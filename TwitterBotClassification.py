from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,roc_curve, auc

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class TwitterBotClassifier:
	def __init__(self, training_dataset, demo_dataset) -> None:

		self.training_dataset = training_dataset
		self.demo_dataset = demo_dataset

		self.test_size = 0.3 #used for splitting dataset into training and testing

		self.bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
					r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
					r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb'\
					r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

	def create_features(self, dataset):
		dataset['screen_name_binary'] = dataset.screen_name.str.contains(self.bag_of_words_bot, case=False, na=False)
		dataset['name_binary'] = dataset.name.str.contains(self.bag_of_words_bot, case=False, na=False)
		dataset['description_binary'] =dataset.description.str.contains(self.bag_of_words_bot, case=False, na=False)
		dataset['status_binary'] = dataset.status.str.contains(self.bag_of_words_bot, case=False, na=False)

		dataset['listed_count_binary'] = (dataset.listed_count>20000)==False

		return dataset

	def initialisation(self) -> None:
		self.training_dataset = self.create_features(self.training_dataset)

		self.features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 
					'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

		self.to_train = self.training_dataset[self.features]
	


	def split_dataset(self) -> None:
		x = self.to_train.iloc[:, :-1]
		y = self.to_train.iloc[:, -1]

		self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size = self.test_size, random_state = 101)


	def train_with_training_dataset(self) -> None:
		decision_tree_classifier =  DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)
		self.dt = decision_tree_classifier.fit(self.xtrain, self.ytrain)

		self.y_pred_train = self.dt.predict(self.xtrain)
		self.y_pred_test = self.dt.predict(self.xtest)

	def print_accuracy(self) -> None:
		print(f'Training Accuracy : {accuracy_score(self.ytrain, self.y_pred_train)}')
		print(f'Testing Accuracy : {accuracy_score(self.ytest, self.y_pred_test)}')

	def plot_roc(self):
		sns.set(font_scale=1.5)
		sns.set_style("whitegrid", {'axes.grid' : False})

		scores_train = self.dt.predict_proba(self.xtrain)
		scores_test = self.dt.predict_proba(self.xtest)

		y_scores_train = []
		y_scores_test = []
		for i in range(len(scores_train)):
			y_scores_train.append(scores_train[i][1])

		for i in range(len(scores_test)):
			y_scores_test.append(scores_test[i][1])
			
		fpr_dt_train, tpr_dt_train, _ = roc_curve(self.ytrain, y_scores_train, pos_label=1)
		fpr_dt_test, tpr_dt_test, _ = roc_curve(self.ytest, y_scores_test, pos_label=1)

		plt.plot(fpr_dt_train, tpr_dt_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
		plt.plot(fpr_dt_test, tpr_dt_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
		plt.title("Decision Tree ROC Curve")
		plt.xlabel("False Positive Rate (FPR)")
		plt.ylabel("True Positive Rate (TPR)")
		plt.legend(loc='lower right')

		plt.show()


	def predict(self):
		self.demo_dataset = self.create_features(self.demo_dataset)
		
		self.to_predict = self.demo_dataset[self.features[:-1]]

		predictions = self.dt.predict(self.to_predict)

		return predictions