# scaled-ml
How to train supervised learning models when your data doesn't fit in memory

## Structure

preprocess.py

```python
import tqdm
import logging

from .src.helpers import get_sample_iterator
from .src.preprocessing import get_preprocessor_class


def parse_args():
    pass


if __name__ == '__main__':
    # setup the program
    program_args, preprocessor_init_args, preprocessor_fit_args = parse_args()
    setup_logging(program_args.log_level)

    # initialize and fit a preprocessor class
    preprocessor_class = get_preprocessor_class(program_args.preprocessor_name)
    preprocessor = preprocessor_class(**vars(preprocessor_init_args))
    preprocessor.fit(program_args.train_dirs, **vars(preprocessor_fit_args))
    
    # preprocess the data to disk
    sample_iterator = get_sample_iterator(train_dirs, val_dirs, test_dirs)
    bar = tqdm.bar(num_iters=sample_iterator.num_samples)
    with mp.pool() as p:
        for (raw_features, label) in sample_iterator:
            p.apply_async(
                preprocessor.sample_to_tfrecord,
                raw_features,
                label,
                program_args.output_dir,
                on_complete=lambda _: bar.update(),
                on_error=lambda error: logging.warning(
                    'Failed to preprocess sample {} with error {}'.format(sample, error)
                )
            )
    
    # save artifacts required to reload the preprocessor
    preprocessor.save(program_args.output_dir)
```

train.py

```python
import json

from .src.training import get_model_class
from .src.helpers import get_sample_iterator, load_preprocessor, get_tensorflow_strategy


def parse_args():
    pass

def setup_logging():
    pass


if __name__ == '__main__':
    # setup the program
    program_args, model_init_args, model_fit_args, model_predict_args = parse_args()
    setup_logging(program_args.log_level)

    # initialize and fit a model to a data directory
    tfrecord_signature_dict = json.loads(program_args.tfrecord_signature_file)
    tensorflow_training_strategy = get_tensorflow_strategy(program_args.strategy_name)
    model_class = get_model_class(program_args.model_name)
    model = model_class(
        tensorflow_training_strategy, 
        tfrecord_signature_file,
        program_args.saved_model_dir,
        **vars(model_init_args)
    )
    model.fit(
        program_args.train_dirs, 
        program_args.eval_dirs,
        **vars(model_fit_args)
    )

    # evaluate the model against the original samples
    preprocessor = load_preprocessor(program_args.preprocessor_dir)
    sample_iterator = get_sample_iterator(program_args.eval_dirs)
    evaluation_metrics = EvaluationMetrics()
    for (raw_features, label) in sample_iterator:
        preprocessed_features = preprocessor.preprocess(raw_features)
        prediction = model.predict(preprocessed_features, **vars(model_predict_args))
        if program_args.pred_dir:
            save_prediction(prediction, label, program_args.pred_dir)
        evaluation_metrics.update(prediction, label)

```
