import tensorflow as tf


def create_model(
    image_width,
    image_height,
    image_channels,
    crop_margin_from_top=80,
    weight_loss_angle=0.8,
    weight_loss_throttle=0.2,
    pre_trained_weights=True,
):
    # Load model with or without pretrained weights
    if pre_trained_weights:
        # Load the MobileNetV2 model
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet"
        )

        # Freeze basemodel weights
        base_model.trainable = False
    else:
        # Load the MobileNetV2 model
        base_model = tf.keras.applications.MobileNetV2(include_top=False)

    # Get image input
    img_in = base_model.get_input_at(0)
    # Get mobilenet output
    x = base_model.get_output_at(0)

    # Add outputs to control steering and acceleration
    angle_out = tf.keras.layers.Dense(units=1, activation="linear", name="angle_out")(x)
    throttle_out = tf.keras.layers.Dense(
        units=1, activation="linear", name="throttle_out"
    )(x)

    # Combine mobilenetv2 base model with new outputs
    model = tf.keras.models.Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss={"angle_out": "mean_squared_error", "throttle_out": "mean_squared_error"},
        loss_weights={
            "angle_out": weight_loss_angle,
            "throttle_out": weight_loss_throttle,
        },
        metrics=["mse", "mae", "mape"],
    )

    return model
