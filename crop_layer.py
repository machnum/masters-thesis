
class CropLayer(object):

    def __init__(self, params, blobs):
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0


    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]

        self.y_start = (input_shape[2] - target_shape[2]) // 2
        self.x_start = (input_shape[3] - target_shape[3]) // 2
        self.y_end = self.y_start + height
        self.x_end = self.x_start + width

        return [[batch_size, num_channels, height, width]]


    def forward(self, inputs):
        return [inputs[0][:,:,self.y_start:self.y_end,self.x_start:self.x_end]]
