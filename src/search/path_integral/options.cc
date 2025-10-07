#include "search/path_integral/options.h"

namespace lczero {

const OptionId kPathIntegralLambdaId{
    {.long_flag = "path-integral-lambda",
     .uci_option = "PathIntegralLambda",
     .help_text =
         "Softmax temperature (lambda) for Path Integral sampling. Lower values increase exploration, higher values increase exploitation.",
     .visibility = OptionId::kAlwaysVisible}};

const OptionId kPathIntegralSamplesId{
    {.long_flag = "path-integral-samples",
     .uci_option = "PathIntegralSamples",
     .help_text =
         "Number of sample paths to generate at root node for Path Integral sampling.",
     .visibility = OptionId::kAlwaysVisible}};

const OptionId kPathIntegralRewardModeId{
    {.long_flag = "path-integral-reward-mode",
     .uci_option = "PathIntegralRewardMode",
     .help_text =
         "Reward calculation mode for quantum_limit sampling: policy (policy head only), cp_score (centipawn only), hybrid (policy * softmax(cp_score)).",
     .visibility = OptionId::kAlwaysVisible}};

const OptionId kPathIntegralModeId{
    {.long_flag = "path-integral-mode",
     .uci_option = "PathIntegralMode",
     .help_text =
         "Path Integral sampling mode: competitive (optimal play with search) or quantum_limit (detailed analysis with policy/value heads).",
     .visibility = OptionId::kAlwaysVisible}};

const OptionId kPathIntegralDebugModeId{
    {.long_flag = "path-integral-debug-mode",
     .uci_option = "PathIntegralDebugMode",
     .help_text =
         "Enable detailed debug logging for Path Integral sampling verification. Outputs structured JSON logs for analysis.",
     .visibility = OptionId::kAlwaysVisible}};

const OptionId kPathIntegralMetricsFileId{
    {.long_flag = "path-integral-metrics-file",
     .uci_option = "PathIntegralMetricsFile",
     .help_text =
         "Output file path for Path Integral debug metrics and logs. If empty, logs are output to stderr only.",
     .visibility = OptionId::kAlwaysVisible}};

}  // namespace lczero

