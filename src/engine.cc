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
#include <cmath>
#include <random>

#include "chess/position.h"
#include "neural/backend.h"
#include "neural/memcache.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "syzygy/syzygy.h"

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
const OptionId kPathIntegralLambda{
    {.long_flag = "path-integral-lambda",
     .uci_option = "PathIntegralLambda",
     .help_text = "The lambda parameter for path integral training.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kPathIntegralSamples{
    {.long_flag = "path-integral-samples",
     .uci_option = "PathIntegralSamples",
     .help_text = "The number of samples to use for path integral training.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kPathIntegralRewardMode{
    {.long_flag = "path-integral-reward-mode",
     .uci_option = "PathIntegralRewardMode",
     .help_text = "The reward mode for path integral training.",
     .visibility = OptionId::kAlwaysVisible}};

}  // namespace

void Engine::PopulateOptions(OptionsParser* options) {
  options->Add<BoolOption>(kPonderId) = false;
  options->Add<StringOption>(kSyzygyTablebaseId);
  options->Add<BoolOption>(kStrictUciTiming) = false;
  options->Add<BoolOption>(kPreload) = false;
  options->Add<FloatOption>(kPathIntegralLambda) = 0.1f;
  options->Get<FloatOption>(kPathIntegralLambda).SetRange(0.001f, 10.0f);
  options->Add<IntOption>(kPathIntegralSamples) = 50;
  options->Get<IntOption>(kPathIntegralSamples).SetRange(1, 100000);
  options->Add<StringOption>(kPathIntegralRewardMode) = "hybrid";
  options->Get<StringOption>(kPathIntegralRewardMode).SetValues({"policy", "cp_score", "hybrid"});
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

  // Path Integral aktifse temperature'ı override et
  if (options_.Get<IntOption>(kPathIntegralSamples) > 0) {
    float temp = 0.7f;
    if (options_.Has("Temperature")) {
      temp = options_.Get<FloatOption>("Temperature");
    }
    const_cast<OptionsDict&>(options_).Set("Temperature", temp);
  }

  if (!backend_ || backend_name != backend_name_ ||
      backend_->UpdateConfiguration(options_) == Backend::NEED_RESTART) {
    backend_name_ = backend_name;
    backend_ = CreateMemCache(BackendManager::Get()->CreateFromParams(options_),
                              options_);
    search_->SetBackend(backend_.get());
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

  // Path Integral Sampling entegrasyonu
  if (options_.Get<IntOption>(kPathIntegralSamples) > 0) {
    auto sampling_results = PathIntegralSample(*last_position_);
    // UCI info satırı ile sampling sonuçlarını ilet
    for (const auto& [move, prob] : sampling_results) {
      std::ostringstream oss;
      oss << "info string path_integral_move " << move.ToUci() << " prob " << prob;
      if (uci_forwarder_) {
        BestMoveInfo info;
        info.info_string = oss.str();
        uci_forwarder_->OutputBestMove(&info);
      }
    }
  }

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

// Path Integral Sampling fonksiyonu
std::vector<std::pair<Move, float>> Engine::PathIntegralSample(const GameState& state) {
  float lambda = options_.Get<FloatOption>(kPathIntegralLambda);
  int samples = options_.Get<IntOption>(kPathIntegralSamples);
  std::string reward_mode = options_.Get<StringOption>(kPathIntegralRewardMode);

  // Root pozisyonunu al
  const Position& position = state.startpos;
  // Legal hamleleri al
  std::vector<Move> legal_moves = position.GetLegalMoves();
  // Policy ve value skorlarını backend'den al
  std::vector<float> policy_scores = backend_->GetPolicyScores(position);
  float value_score = backend_->GetValueScore(position);

  std::vector<float> scores;
  std::vector<Move> moves;
  for (size_t i = 0; i < legal_moves.size(); ++i) {
    float policy = (i < policy_scores.size()) ? policy_scores[i] : 0.0f;
    float cp_score = value_score;
    float score = 0.0f;
    if (reward_mode == "policy") score = policy;
    else if (reward_mode == "cp_score") score = cp_score;
    else if (reward_mode == "hybrid") score = policy * std::exp(lambda * cp_score);
    moves.push_back(legal_moves[i]);
    scores.push_back(score);
  }
  // Softmax (log-sum-exp)
  float max_score = *std::max_element(scores.begin(), scores.end());
  std::vector<float> arr_scaled;
  for (float s : scores) arr_scaled.push_back((s - max_score) * lambda);
  float sum_exp = 0.0f;
  for (float v : arr_scaled) sum_exp += std::exp(v);
  float log_sum_exp = std::log(sum_exp);
  std::vector<float> probs;
  for (float v : arr_scaled) probs.push_back(std::exp(v - log_sum_exp));
  // Güvenlik: nan veya negatif toplam varsa eşit olasılık
  float total_prob = 0.0f;
  for (float p : probs) total_prob += p;
  if (std::isnan(total_prob) || total_prob <= 0.0f) {
    float eq = 1.0f / probs.size();
    for (size_t i = 0; i < probs.size(); i++) probs[i] = eq;
  }
  // Sonuçları döndür
  std::vector<std::pair<Move, float>> result;
  for (size_t i = 0; i < moves.size(); ++i) {
    result.push_back({moves[i], probs[i]});
  }
  return result;
}

}  // namespace lczero
