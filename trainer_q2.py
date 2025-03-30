import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

train_data_path = "data/train_data_SU_A2_q2.csv"
test_size = 0.2


def main():
    """Main module to train"""
    # read train data
    df_train_data = pd.read_csv(train_data_path)

    # train test split
    df_train, df_test = train_test_split(df_train_data, test_size=test_size)

    # y for both train and test
    y_train = df_train.pop("class_label")
    y_test = df_test.pop("class_label")

    # Normalize the dataset
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)

    # define cls model
    cls_model = RandomForestClassifier(
        n_estimators=100, max_depth=18, min_samples_split=20, random_state=1234
    )
    cls_model.fit(df_train, y_train)

    # check accuracy of model
    print("###################### Model Evaluation ######################\n")
    score = accuracy_score(y_train, cls_model.predict(df_train))
    print("Train accuracy: {}".format(score))
    score = accuracy_score(y_test, cls_model.predict(df_test))
    print("Test accuracy: {}".format(score))
    score = f1_score(y_train, cls_model.predict(df_train), average="weighted")
    print("Train f1 score: {}".format(score))
    score = f1_score(y_test, cls_model.predict(df_test), average="weighted")
    print("Test f1 score: {}".format(score))

    # save model
    filename = 'models/classify_language.sav'
    pickle.dump(cls_model, open(filename, 'wb'))


if __name__ == "__main__":
    main()