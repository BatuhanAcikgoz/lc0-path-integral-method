/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "engine.h"

#include <algorithm>
#include "chess/board.h"  // Taş sabitleri için
#include "utils/string.h" // SquareToString için (muhtemelen bu dosyada)
#include "chess/position.h"
#include "chess/callbacks.h"
#include "neural/backend.h"
#include "neural/memcache.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "syzygy/syzygy.h"
#include "utils/optionsparser.h"
#include "chess/types.h"
#ifdef USE_PATH_INTEGRAL
#include "search/path_integral/controller_simple.h"
#include "search/path_integral/interfaces.h"
#include "search/path_integral/options.h"
#endif

namespace lczero {
namespace {
const OptionId kSyzygyTablebaseId{
    {.long_flag = "syzygy-paths",
     .uci_option = "SyzygyPath",
     .help_text =
         "List of Syzygy tablebase directories, list entries separated by "
         "system separator (\";\" for Windows, \":\" for Linux).",
     .short_flag = 's',
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kStrictUciTiming{
    {.long_flag = "strict-uci-timing",
     .uci_option = "StrictTiming",
     .help_text = "The UCI host compensates for lag, waits for the 'readyok' "
                  "reply before sending 'go' and only then starts timing.",
     .visibility = OptionId::kProOnly}};
const OptionId kPonderId{
    {.long_flag = "",
     .uci_option = "Ponder",
     .help_text =
         "Indicates to the engine that it will be requested to ponder. This "
         "postpones resetting the search tree until the search is started.",
     .visibility = OptionId::kAlwaysVisible}};

const OptionId kPreload{"preload", "",
                        "Initialize backend and load net on engine startup."};

}  // namespace

// Path Integral UCI option definitions with external linkage
// (defined in search/path_integral/options.cc)

void Engine::PopulateOptions(OptionsParser* options) {
  options->Add<BoolOption>(kPonderId) = false;
  options->Add<StringOption>(kSyzygyTablebaseId);
  options->Add<BoolOption>(kStrictUciTiming) = false;
  options->Add<BoolOption>(kPreload) = false;
  
#ifdef PATH_INTEGRAL_ENABLED
  // Register Path Integral UCI options
  options->Add<FloatOption>(kPathIntegralLambdaId, 0.001f, 10.0f) = 0.1f;
  options->Add<IntOption>(kPathIntegralSamplesId, 1, 100000) = 50;
  
  std::vector<std::string> reward_modes = {"policy", "cp_score", "hybrid"};
  options->Add<ChoiceOption>(kPathIntegralRewardModeId, reward_modes) = "hybrid";
  
  std::vector<std::string> sampling_modes = {"competitive", "quantum_limit"};
  options->Add<ChoiceOption>(kPathIntegralModeId, sampling_modes) = "competitive";
#endif
}

namespace {
GameState MakeGameState(const std::string& fen,
                        const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  ChessBoard cur_board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    Move m = cur_board.ParseMove(move);
    state.moves.push_back(m);
    cur_board.ApplyMove(m);
    cur_board.Mirror();
  }
  return state;
}
}  // namespace

class Engine::UciPonderForwarder : public UciResponder {
 public:
  UciPonderForwarder(Engine* engine) : engine_(engine) {}

  void OutputBestMove(BestMoveInfo* info) override {
    if (!wrapped_) return;
    wrapped_->OutputBestMove(info);
  }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    if (!wrapped_) return;
    if (engine_->last_go_params_ && engine_->last_go_params_->ponder) {
      assert(engine_->last_position_ &&
             !engine_->last_position_->moves.empty());
      const Move ponder_move_ = engine_->last_position_->moves.back();
      // Output all stats from main variation (not necessary the ponder move)
      // but PV only from ponder move.
      ThinkingInfo ponder_info;
      for (const auto& info : *infos) {
        if (info.multipv <= 1) {
          ponder_info = info;
          if (ponder_info.mate) ponder_info.mate = -*ponder_info.mate;
          if (ponder_info.score) ponder_info.score = -*ponder_info.score;
          if (ponder_info.depth > 1) ponder_info.depth--;
          if (ponder_info.seldepth > 1) ponder_info.seldepth--;
          if (ponder_info.wdl)
            std::swap(ponder_info.wdl->w, ponder_info.wdl->l);
          ponder_info.pv.clear();
        }
        if (!info.pv.empty() && info.pv[0] == ponder_move_) {
          ponder_info.pv.assign(info.pv.begin() + 1, info.pv.end());
        }
      }
      infos->clear();
      infos->push_back(ponder_info);
    }
    wrapped_->OutputThinkingInfo(infos);
  }

  void Register(UciResponder* wrapped) {
    if (wrapped_) {
      throw Exception("UciPonderForwarder already has a wrapped responder");
    }
    wrapped_ = wrapped;
  }
  void Unregister(UciResponder* wrapped) {
    if (wrapped_ != wrapped) {
      throw Exception("UciPonderForwarder doesn't have this wrapped responder");
    }
    wrapped_ = nullptr;
  }

 private:
  UciResponder* wrapped_ = nullptr;
  Engine* const engine_;
};

Engine::Engine(const SearchFactory& factory, const OptionsDict& opts)
    : uci_forwarder_(std::make_unique<UciPonderForwarder>(this)),
      options_(opts),
      search_(factory.CreateSearch(uci_forwarder_.get(), &options_)) {
#ifdef USE_PATH_INTEGRAL
  // Initialize Path Integral controller (backend will be set later in UpdateBackendConfig)
  path_integral_controller_ = std::make_unique<SimplePathIntegralController>(options_);
#endif
  
  if (options_.Get<bool>(kPreload)) {
    UpdateBackendConfig();
    EnsureSyzygyTablebasesLoaded();
  }
}

Engine::~Engine() { EnsureSearchStopped(); }

void Engine::EnsureSearchStopped() {
  search_->AbortSearch();
  search_->WaitSearch();
}

void Engine::UpdateBackendConfig() {
  LOGFILE << "Update backend configuration.";
  const std::string backend_name =
      options_.Get<std::string>(SharedBackendParams::kBackendId);
  if (!backend_ || backend_name != backend_name_ ||
      backend_->UpdateConfiguration(options_) == Backend::NEED_RESTART) {
    backend_name_ = backend_name;
    backend_ = CreateMemCache(BackendManager::Get()->CreateFromParams(options_),
                              options_);
    search_->SetBackend(backend_.get());
    
#ifdef USE_PATH_INTEGRAL
    // Update Path Integral controller with new backend
    if (path_integral_controller_) {
      path_integral_controller_ = std::make_unique<SimplePathIntegralController>(options_, backend_.get());
    }
#endif
  } else {
    backend_->SetCacheSize(
        options_.Get<int>(SharedBackendParams::kNNCacheSizeId));
  }
}

void Engine::EnsureSyzygyTablebasesLoaded() {
  const std::string tb_paths = options_.Get<std::string>(kSyzygyTablebaseId);
  if (tb_paths == previous_tb_paths_) return;
  previous_tb_paths_ = tb_paths;

  if (tb_paths.empty()) {
    LOGFILE << "Reset Syzygy tablebases.";
    syzygy_tb_.reset();
  } else {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_.reset();
    }
  }

  search_->SetSyzygyTablebase(syzygy_tb_.get());
}

// Initializes the search with either the specified position for the normal
// search or the position one ply trimmed for the ponder search.
void Engine::InitializeSearchPosition(bool for_ponder) {
  LOGFILE << "Setting a new search position.";
  assert(last_position_);
  if (!for_ponder) {
    search_->SetPosition(*last_position_);
    return;
  }
  if (last_position_->moves.empty()) {
    throw Exception("Ponder search requires at least one move.");
  }
  GameState position = *last_position_;
  position.moves.pop_back();
  search_->SetPosition(position);
  return;
}

void Engine::SetPosition(const std::string& fen,
                         const std::vector<std::string>& moves) {
  EnsureSearchStopped();
  ponder_enabled_ = options_.Get<bool>(kPonderId);
  strict_uci_timing_ = options_.Get<bool>(kStrictUciTiming);
  if (!strict_uci_timing_) search_->StartClock();
  UpdateBackendConfig();
  EnsureSyzygyTablebasesLoaded();
  last_position_ = MakeGameState(fen, moves);
  if (!ponder_enabled_) InitializeSearchPosition(/*for_ponder=*/false);
}

void Engine::NewGame() {
  if (backend_) backend_->ClearCache();
  search_->NewGame();
  
#ifdef USE_PATH_INTEGRAL
  // Update Path Integral options in case they changed
  if (path_integral_controller_) {
    path_integral_controller_->UpdateOptions(options_);
  }
#endif
  
  SetPosition(ChessBoard::kStartposFen, {});
}


void Engine::Go(const GoParams& params) {
  if (!ponder_enabled_ && params.ponder) {
    throw Exception(
        "Ponder is not enabled, but the ponder search is requested.");
  }
  if (strict_uci_timing_) search_->StartClock();
  if (!last_position_) NewGame();
  if (ponder_enabled_) InitializeSearchPosition(params.ponder);
  last_go_params_ = params;

#ifdef USE_PATH_INTEGRAL
  // Path Integral integration - currently in development
  // For now, let LC0's standard search handle move selection
  // Path Integral will be integrated into the search process in future versions
  
  if (path_integral_controller_ && path_integral_controller_->IsEnabled()) {
    LOGFILE << "Path Integral: Enabled but delegating to LC0 search system";
    
    // TODO: Integrate Path Integral sampling into LC0's search tree
    // This will require modifying the MCTS algorithm to use Path Integral sampling
    // at the root node while maintaining LC0's search quality
  }
#endif

  // Standard LC0 search (fallback or when Path Integral is disabled)
  search_->StartSearch(params);
}

void Engine::Wait() { search_->WaitSearch(); }

void Engine::Stop() { search_->StopSearch(); }

void Engine::PonderHit() {
  if (!last_go_params_ || !last_go_params_->ponder) {
    throw Exception("ponderhit while not pondering");
  }
  EnsureSearchStopped();
  search_->StartClock();
  last_go_params_->ponder = false;
  InitializeSearchPosition(/*ponder=*/false);
  search_->StartSearch(*last_go_params_);
}

void Engine::RegisterUciResponder(UciResponder* responder) {
  uci_forwarder_->Register(responder);
}

void Engine::UnregisterUciResponder(UciResponder* responder) {
  uci_forwarder_->Unregister(responder);
}

}  // namespace lczero
