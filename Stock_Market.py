import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
# print(sp500)
# print(type(sp500))

# -------------------------Plotting using Matplotlib--------------------------
y = np.array(sp500["Close"])
x = np.array(sp500.index)

plt.ylabel("Closing Prices ($)")
plt.xlabel("Date")
plt.plot(x, y)
plt.savefig("Closing_Prices.png")
plt.show(block=False)
plt.close()
# -------------------------Plotting using Matplotlib--------------------------

del sp500["Dividends"]  # Deleting the column which we will not use
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)  # Tomorrow column gives the true value of the stock price for the next day
# print(sp500)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)   # astype here is converting bool to int
# print(sp500)

sp500 = sp500.loc["1990-01-01":].copy()   # Loc locates the data with index 1990-01-01
# and :___ after that tells loc to find all the data with index after 1990-01-01

# print(sp500)

# ----------------------------------------Model-------------------------------------------
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
# print(train)
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]


def predict(train_data, test_data, predictors_column_names, model_used):
    model_used.fit(train_data[predictors_column_names], train_data["Target"])

    predicted_data = model_used.predict_proba(test_data[predictors_column_names])[:, 1]
    # .predict_proba tells us about the probability whether the price will go up/down tomorrow
    # [:, 1] fetches only the second column of proba, which tells us the probability the price will go up tomorrow
    predicted_data[predicted_data >= 0.6] = 1
    predicted_data[predicted_data < 0.6] = 0
    predicted_data = pd.Series(data=predicted_data, index=test_data.index, name="Predictions")

    combined_data = pd.concat([test_data["Target"], predicted_data], axis=1)
    # print(combined)

    return combined_data


def backtest(data, model_used, predictors_column_names, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        # The shape attribute for numpy arrays returns the
        # dimensions of the array.If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
        train_data = data.iloc[:i].copy()
        test_data = data.iloc[i:(i + step)].copy()
        predicted_data = predict(train_data=train_data, test_data=test_data,
                                 predictors_column_names=predictors_column_names, model_used=model_used)
        all_predictions.append(predicted_data)
    return pd.concat(all_predictions)


# precision, combined = predict(train_data=train, test_data=test, predictors_column_names=predictors, model_used=model)
# predictions = backtest(data=sp500, model_used=model, predictors_column_names=predictors)
# print(predictions["Predictions"].value_counts())

# combined = predict(train_data=train, test_data=test, predictors_column_names=predictors, model_used=model)
# print(combined)

# precision = precision_score(y_true=predictions["Target"], y_pred=predictions["Predictions"])
# print(precision)

rolling_avg_horizon = [2, 5, 60, 250, 1000]
new_predictors = []
for i in rolling_avg_horizon:
    rolling_avg = sp500.rolling(i).mean()
    ratio_column = f"Close Ratio {i}"
    sp500[ratio_column] = sp500["Close"]/rolling_avg["Close"]
    trend_column = f"Trend {i}"
    sp500[trend_column] = sp500.shift(1).rolling(i).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()
# print(sp500)

predictions = backtest(model_used=model, predictors_column_names=new_predictors, data=sp500)
print(predictions.value_counts())
precision = precision_score(y_true=predictions["Target"], y_pred=predictions["Predictions"])
print(precision)

# -----------------------------Plotting to see how well our predictions match with targets------------------------------
x = np.array(predictions.index)
y1 = np.array(predictions["Target"])
y2 = np.array(predictions["Predictions"])

plt.ylabel("Closing Prices ($)")
plt.xlabel("Date")
plt.plot(x, y1, label='Targets')
plt.plot(x, y2, label='Predictions')
plt.legend(loc='upper left', fontsize='medium')
plt.savefig("Final_Prediction.png")
plt.show(block=False)
plt.close()
# -----------------------------Plotting to see how well our predictions match with targets------------------------------
