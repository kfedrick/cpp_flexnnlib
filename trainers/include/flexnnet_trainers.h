//
// Created by kfedrick on 10/5/19.
//

#ifndef FLEX_NEURALNET_FLEXNNET_TRAINER_H_
#define FLEX_NEURALNET_FLEXNNET_TRAINER_H_

typedef std::tuple<double, std::valarray<double>> NNErrorTyp;
typedef NNErrorTyp
   (*ErrorFunc)(const std::valarray<double>& _target, const std::valarray<double>& _actual);

enum TrainingStopSignal
{
   MAX_EPOCHS_REACHED, CRITERIA_MET, MAX_FAILBACK_REACHED
};

#endif //FLEX_NEURALNET_FLEXNNET_TRAINER_H_
