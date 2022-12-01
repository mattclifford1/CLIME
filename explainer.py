import fatf.transparency.predictions.surrogate_explainers as surrogates

def LIME(data, clf):
    lime = surrogates.TabularBlimeyLime(data['X'], clf)
    return lime


if __name__ == '__main__':
    import data_generation
    import train_model

    # get dataset
    train_data, test_data = data_generation.get_data()

    # train model
    clf = train_model.SVM(train_data)

    # get lime explainer
    lime = LIME(train_data, clf)

    lime_explanation = lime.explain_instance(test_data['X'][0, :], samples_number=500)
    print(lime_explanation)
