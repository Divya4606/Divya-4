model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

model.fit(X_train_scaled, y_train,
          early_stopping_rounds=10,
          eval_set=[(X_test_scaled, y_test)],
          verbose=False)

y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost RMSE: {rmse:.2f}")
