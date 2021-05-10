#https://github.com/deepset-ai/FARM/blob/master/examples/text_pair_classification.py

import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.eval import Evaluator

def text_pair_classification(lang_model="ProsusAI/finbert", sdir="saved_models/finbert-reddit"):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    # ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_text_pair_classification")

    ##########################
    ########## Settings ######
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 4
    lr = 2e-5
    batch_size = 16
    evaluate_every = 100
    label_list = ['0', '1', '2']


    save_dir = Path(sdir)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset.
    # The TextPairClassificationProcessor expects a csv with columns called "text', "text_b" and "label"
    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                                label_list=label_list,
                                                metric="f1_macro",
                                                max_seq_len=384,
                                                train_filename="sentiment_train.tsv",
                                                test_filename="sentiment_test.tsv",
                                                dev_split=0,
                                                data_dir=Path("data/SentimentData"))

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        # max_processes=1,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task
    prediction_head = TextClassificationHead(num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=lr,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluator_test=False,
        device=device)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Test Model and return results

    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    results = evaluator.eval(model)

    return results[0]["report"]


if __name__ == "__main__":

    pairs = [
        ('ProsusAI/finbert', 'saved_models/finbert-reddit'),
        ('saved_models/finbert-finetuned', 'saved_models/finbert-pretrained-reddit'),
        ('bert-base-cased', 'saved_models/bert-base-cased-reddit'),
        ('saved_models/bert-english-lm-tutorial', 'saved_models/bert-base-cased-pretrained-reddit')
    ]
    reports = []
    for x, y in pairs:
        r = text_pair_classification(lang_model=x, sdir=y)
        reports.append((x, r))

    for x, y in reports:
        print(x)
        print(y)