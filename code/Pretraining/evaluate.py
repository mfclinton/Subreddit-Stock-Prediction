# https://github.com/deepset-ai/FARM/blob/master/examples/evaluation.py

from farm.utils import initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor, SquadProcessor
from farm.data_handler.data_silo import DataSilo
from farm.eval import Evaluator
from farm.modeling.adaptive_model import AdaptiveModel
from pathlib import Path

def evaluate_classification():
    ##########################
    ########## Settings
    ##########################
    device, n_gpu = initialize_device_settings(use_cuda=True)
    do_lower_case = False
    batch_size = 16

    data_dir = Path("data/SentimentData")
    evaluation_filename = "sentiment_test.tsv"
    metric = "f1_macro"
    lang_model = "cardiffnlp/twitter-roberta-base-sentiment"
    label_list = ['2', '1', '0']

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset

    processor = TextClassificationProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        train_filename=None,
        dev_filename=None,
        dev_split=0,
        test_filename=evaluation_filename,
        data_dir=data_dir,
    )

    # 3. Create a DataSilo that loads dataset, provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an Evaluator
    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )

    # 5. Load model
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="text_classification")
    # use "load" if you want to use a local model that was trained with FARM
    # model = AdaptiveModel.load(lang_model, device=device)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    # 6. Run the Evaluator
    results = evaluator.eval(model)
    f1_score = results[0]["f1_macro"]
    print("Macro-averaged F1-Score:", f1_score)
    print(results[0]["report"])


if __name__ == "__main__":
    evaluate_classification()