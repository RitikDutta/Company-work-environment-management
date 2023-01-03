import numpy as np
from data_processing.converter import Converter

class Prediction:
    """
               This class shall be used for predictions of landmarks of data.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

               """
def predict(landmark):
        """
                Method Name: predict
                Description: This method predict the class from landmark data.
                Output: None
                On Failure: Raise Exception

                Written By: Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
    x = convert_mp_to_dataframe(landmark)
    x = pd.DataFrame(x.iloc[0:].values.reshape(1, -1))
    x = x.apply(pd.to_numeric, errors='coerce', downcast='float')
    class_labels = {0: 'away', 1: 'phone', 2: 'working'}

    # Make a prediction
    prediction = model.predict(x)

    # Find the index of the highest probability
    class_index = np.argmax(prediction)

    # Look up the class label in the dictionary
    class_label = class_labels[class_index]

    # Print the class label
#     print("Class label:", class_label)
    return class_label 
