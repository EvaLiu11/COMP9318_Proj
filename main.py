import helper
import submission

training_data = helper.read_data('./asset/training_data.txt')
test_data = helper.read_data('./asset/test_words.txt')
classifier_path = './asset/Logistic.dat'
submission.train(training_data, classifier_path,multi=True,DEBUG=True)
submission.test(test_data,classifier_path,DEBUG=True)