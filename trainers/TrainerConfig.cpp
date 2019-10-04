//
// Created by kfedrick on 7/29/19.
//

#include "TrainerConfig.h"

using flexnnet::TrainerConfig;

TrainerConfig::TrainerConfig()
{
   set_batch_size (DEFAULT_BATCH_MODE);
   set_max_epochs (DEFAULT_MAX_EPOCHS);
   set_error_goal (DEFAULT_ERROR_GOAL);
   set_display_frequency (DEFAULT_DISPLAY_FREQ);
   set_report_frequency (DEFAULT_REPORT_FREQ);
   set_max_validation_failures (DEFAULT_MAX_VALIDATION_FAIL);
   set_verbose (DEFAULT_VERBOSE);
}
