import yaml
import argparse
import wandb
import os
from mobilenet import MobileNetV3Small_Raw
from tensorflow.keras.callbacks import ModelCheckpoint
from models import build_tf_model
from sklearn import model_selection
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from dataset.tf_data_loader import get_datasets, create_tf_dataset
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import load_model


def calculate_precision_and_recall(model, dataset, config):
    y_true = []
    y_pred = []

    for batch in dataset:
        X_batch, y_batch = batch
        if config["dataset"]["preprocess"] == []:
            # Assuming X_batch has shape (16, 1, 16000)---> Reshape to (16, 16000, 1)
            X_batch = tf.reshape(X_batch, [X_batch.shape[0], config["dataset"]["sample_rate"], 1])

        logits = model.predict(X_batch)
        preds = np.argmax(logits, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute precision and recall
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return precision, recall

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    np.random.seed(config["seed"])

    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu_id"]

    print("creating training and validation dataset")
    
    # Get data loaders
    train_val_files, _, test_datasets = get_datasets(config)

    #spliting to get 80% train and 10% val and 10% test of total data
    train_files, val_files = model_selection.train_test_split(train_val_files, test_size=0.11, random_state=config["seed"],stratify=[item for item in train_val_files.MMSI])

    if config["dataset"]["augment"]:
        train_original = create_tf_dataset(
            root_dir=config["dataset"]["root_dir"],
            files_df=train_files,
            transform=config["dataset"]["preprocess"],
            target_sample_rate=config["dataset"]["sample_rate"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=True
            )
        
        train_augmented = create_tf_dataset(
            root_dir=config["dataset"]["root_dir"],
            files_df=train_files,
            transform=config["dataset"]["preprocess"],
            target_sample_rate=config["dataset"]["sample_rate"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            apply_augmentation=True,
            augmentation_type=config["dataset"]["augmentation_type"]
        )
        
        train_dataset = train_original.concatenate(train_augmented)
        val_original = create_tf_dataset(
            root_dir=config["dataset"]["root_dir"],
            files_df=val_files,
            transform=config["dataset"]["preprocess"],
            target_sample_rate=config["dataset"]["sample_rate"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=False
        )
        val_augmented = create_tf_dataset(
            root_dir=config["dataset"]["root_dir"],
            files_df=val_files,
            transform=config["dataset"]["preprocess"],
            target_sample_rate=config["dataset"]["sample_rate"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=False,
            apply_augmentation=True,
            augmentation_type=config["dataset"]["augmentation_type"]
        )
        val_dataset = val_original.concatenate(val_augmented)
    else:
        train_dataset = create_tf_dataset(
            root_dir=config["dataset"]["root_dir"],
            files_df=train_files,
            transform=config["dataset"]["preprocess"],
            target_sample_rate=config["dataset"]["sample_rate"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=True
        )
        
        val_dataset = create_tf_dataset(
            root_dir=config["dataset"]["root_dir"],
            files_df=val_files,
            transform=config["dataset"]["preprocess"],
            target_sample_rate=config["dataset"]["sample_rate"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=False
        )

    # Build model    
    model = build_tf_model(config)
    if model is None:
        return
    
    run_name = config["wandb"]["run_name"]
    
    # Initialize wandb
    if config["wandb"]["log"]:
        os.environ["WANDB_API_KEY"] = config["wandb"]["key"]
        if config["resume"]:
            wandb.init(
                project=config["wandb"]["project"],
                id=config["wandb"]["id"],
                tags=config["wandb"]["tags"],
                config=config,
                resume="must"
            )
        else:
            wandb.init(
                project=config["wandb"]["project"],
                name=run_name,
                tags=config["wandb"]["tags"],
                config=config
            )
            
    if config["resume"]:
        model.load_weights(config["resume"])
        initial_epoch = int(config["resume"].split("model-")[1].split(".keras")[0])  # Extract epoch from filename

    best_checkpoint_path = os.path.join(config["output_dir"], 'models', run_name, 'best_model.keras')
    
    # Checkpointing
    best_checkpoint_callback = ModelCheckpoint(
        filepath=best_checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False
    )
 
    model_callback = ModelCheckpoint(
        filepath=os.path.join(config["output_dir"], 'models', run_name, 'model-{epoch}.keras'),
        save_best_only=False,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False
    )

    # Construct the full directory path where models will be saved
    model_save_dir = os.path.join(config["output_dir"], 'models', run_name)

    # Create the directory if it doesn't exist
    # This will create all intermediate directories if they are missing
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Ensured model saving directory exists: {model_save_dir}")

    wandb_checkpoint_callback = WandbModelCheckpoint(
        filepath=os.path.join(config["output_dir"], 'models', run_name, 'model.keras'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
    ) if config["wandb"]["log"] else None

    # Training
    if config["wandb"]["log"]:
        callbacks = [WandbMetricsLogger(), wandb_checkpoint_callback, best_checkpoint_callback, model_callback]
    else:
        callbacks = [best_checkpoint_callback, model_callback]
    
    model.fit(
        train_dataset,
        initial_epoch=0 if not config["resume"] else initial_epoch,
        epochs=config["hyperparameters"]["epochs"],
        validation_data=val_dataset,
        callbacks=callbacks
    )
    # Load the best model and perform inference on it
    best_checkpoint_path = os.path.join(config["output_dir"], 'models', run_name, 'best_model.keras')
    best_model = load_model(best_checkpoint_path)

    print(f"Evaluating on test sets:")
    if config["wandb"]["log"]:
        if len(test_datasets) > 0: 
            
            for i, test_set in enumerate(test_datasets):
                test_results = best_model.evaluate(test_set, callbacks=callbacks)
                precision, recall = calculate_precision_and_recall(best_model, test_set, config)
                
                wandb.log({f"test/dataloader_{i}/loss": test_results[0], f"test/dataloader_{i}/accuracy": test_results[1]})
                wandb.log({f"test/dataloader_{i}/precision": precision, f"test/dataloader_{i}/recall": recall})
    
        wandb.finish()
    else:
        # Perform evaluations even if not logging to wandb
        if len(test_datasets) > 0: 
            for i, test_set in enumerate(test_datasets):
                test_results = best_model.evaluate(test_set, callbacks=callbacks)
                precision, recall = calculate_precision_and_recall(best_model, test_set, config)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a TensorFlow model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
