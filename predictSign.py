import numpy as np
from keras import backend as K
import operator
import json

K.set_image_data_format('channels_first')  # Set the image data format

# Change output array according to data
output = ["House", "Aboard", "Baby", "Bowl", "Friend", "IorMe", "Money", "Opposite", "Prisoner", "You"]
def predictSign(frame, model):
    global output
    image = np.array(frame).flatten()
    img_channels, img_x, img_y = model.input_shape[1:]  # Extract input shape from the model
    image = image.reshape(img_channels, img_x, img_y)
    image = image.astype('float32')
    image = image / 255
    image = image.reshape(1, img_channels, img_x, img_y)
    prob_array = model.predict(image)  # Use predict method instead of predict_proba
    
    prob_map = {}
    for i, label in enumerate(output):
        prob_map[label] = prob_array[0][i] * 100
        
    guess = max(prob_map.items(), key=operator.itemgetter(1))[0]
    prob = prob_map[guess]

    if prob > 5.0:
        with open('output.txt', 'w') as outfile:
            json.dump(prob_map, outfile)
        print(str(guess) + " " + str(prob))
        return str(guess)
    else:
        return "No Output"

# Define the frame and model variables
# Example frame (replace this with your actual frame or image data)
frame = np.random.rand(200, 200)  # Example random frame
# Example model (replace this with your actual trained model)
model = create_cnn_model()  # Replace 'create_cnn_model()' with your actual trained model

# Call the predictSign function with the frame and model
predicted_sign = predictSign(frame, model)
print("Predicted sign:", predicted_sign)
