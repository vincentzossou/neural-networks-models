from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, Activation


class UNETModel(object):

	def __init__(self):
		print("UNET Model initialized")

	def conv_block(self, input, num_filters):
		x = Conv2D(num_filters, 3, padding="same")(input)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)

		x = Conv2D(num_filters, 3, padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		
		return x

	def encoder_block(self, input, num_filters):
		x = self.conv_block(input, num_filters)
		p = MaxPool2D((2, 2))(x)
		return x, p

	def decoder_block(self, input, skip_features, num_filters):
	    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
	    x = Concatenate()([x, skip_features])
	    x = self.conv_block(x, num_filters)
	    return x

	def build_unet(self, input_shape, num_class=1):
	    inputs = Input(input_shape)

	    s1, p1 = self.encoder_block(inputs, 64)
	    s2, p2 = self.encoder_block(p1, 128)
	    s3, p3 = self.encoder_block(p2, 256)
	    s4, p4 = self.encoder_block(p3, 512)

	    b1 = self.conv_block(p4, 1024)

	    d1 = self.decoder_block(b1, s4, 512)
	    d2 = self.decoder_block(d1, s3, 256)
	    d3 = self.decoder_block(d2, s2, 128)
	    d4 = self.decoder_block(d3, s1, 64)

	    outputs = Conv2D(num_class, 1, padding="same", activation="sigmoid")(d4)

	    model = Model(inputs, outputs, name="U-Net")
	    return model
