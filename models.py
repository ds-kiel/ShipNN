
import tensorflow as tf
from tensorflow.keras import losses
from resnet import ResNet
from mobilenet import MobileNetV3Small_Raw

def build_tf_model(config):
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    in_channels = len(config["dataset"]["preprocess"]) if config["dataset"]["preprocess"] else 1

    if model_name == "mobilenetv3":
        if config["dataset"]["preprocess"]:
            model = tf.keras.applications.MobileNetV3Small(
                input_shape=(config["dataset"]["FREQ_BINS"], config["dataset"]["HOP_LENGTH"], in_channels),
                include_preprocessing=False,
                include_top=True,
                classes=num_classes,
                weights=None,
                alpha=config["model"]["width"],
                dropout_rate=config["hyperparameters"]["dropout"],
            )
            
        else:
            model= MobileNetV3Small_Raw(config=config,
                                        include_top=True, 
                                        classes=num_classes, 
                                        alpha=config["model"]["width"], 
                                        dropout_rate=config["hyperparameters"]["dropout"],
                                        weights=None)
            

    elif model_name == "resnet18":
        model = ResNet(config = config, classes=num_classes)
        
    elif "mcunet" in model_name:
        # MCUNet conversion is complex and will be handled separately.
        # For now, returning a placeholder.
        print(f"MCUNet model ({model_name}) conversion is not yet implemented.")
        return None
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["hyperparameters"]["learning_rate"])
    loss_function = losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[
            'accuracy'
        ]
    )
    
    return model
