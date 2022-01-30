import numpy as np

outputs=np.load('outputs.npy',allow_pickle=True)
array_for_train_data=np.load('array_for_10_trainwindows.npy',allow_pickle=True)
outputs=outputs.tolist()
array_for_train_data=array_for_train_data.tolist()
print(outputs[0][0])
"""
    api:
        array_for_train_data - same structure with array_for_LSTM, except brand with comments number that less than or equals to trainwindows deleted.
            Example:
                print(array_for_train_data[0][0])
                output:
                    [['really', 'like', 'microwave', 'like', 'easy', 'get', 'amazon', 'really', 'good', 'convection', 'oven', 'although', 'somewhat', 'noisy', 'fan', 
                    'easy', 'install', 'came', 'clear', 'instructions', 'amazon', 'delivered', 'quickly', 'best', 'price', 'anywhere'], 
                    5, 6, 10, 'Y', Timestamp('2007-01-04 00:00:00'), 0]
            Note:
                none

        outputs - almost same structure with array_for_LSTM
            outputs[i] - all n predicted comments for ith brand (n = train_windows, i here is the brand order in array_for_train_data rather than the brand order in array_for_LSTM)
            for every item it looks like 
            [[string containing all the words except for stop words],star rating,total votes,verified purchase(1-Yes;0-No),diff_date(in days)]
            Example:
                print(outputs[0][0])
                output:
                    [['mount', 'models', 'paid', 'looking', 'mechanism', 'keypad', 'newly', 'frozen', 'cu', 'close', 'come', 'crazy', 'blue', 'cabinet', 'chat', '58', 
                    '00', '125', '29', '2015', '150', '3rd', '11', '04', '29', '300', '26', '220v', '11', '1200', '2nd', '27', '2014', '14', '17', 'additionally', '23'],
                    3.5709089040756226, 22.98689065501094, 37.08765381574631, 0.7639150619506836, 44.649807915091515]
            Note:
                If (dtype = integer) is preferred, just let me know

"""