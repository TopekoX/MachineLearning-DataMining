import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
cv = KFold(5, shuffle=True, random_state=SEED)


def cv_rmse(model):
    return -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()


def train_rmse(model):
    return np.sqrt(mean_squared_error(y_train, model.fit(X_train, y_train).predict(X_train)))


# 1. KNN Regressor: prediksi = rata-rata target k tetangga terdekat
print("KNN k   RMSE latih  RMSE CV")
for k in [1, 5, 10, 20, 40]:
    m = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    print(f"{k:<7} {train_rmse(m):<10.2f}  {cv_rmse(m):.2f}")

# 2. Decision Tree Regressor: prediksi = rata-rata target di daun
print("\nTree max_depth  RMSE latih  RMSE CV  jumlah daun")
for depth in [1, 2, 3, 5, None]:
    m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, random_state=SEED)
    leaves = m.fit(X_train, y_train).get_n_leaves()
    print(f"{str(depth):<15} {train_rmse(m):<10.2f}  {cv_rmse(m):<7.2f}  {leaves}")

# 3. Random Forest Regressor: rata-rata banyak pohon
print("\nRF n_estimators  RMSE CV")
for n in [1, 10, 100, 300]:
    m = RandomForestRegressor(n_estimators=n, min_samples_leaf=5, random_state=SEED)
    print(f"{n:<16} {cv_rmse(m):.2f}")

# 4. Evaluasi uji untuk satu pilihan per model
models = {
    "KNN (k=20)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=20)),
    "Tree (depth=3)": DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED),
}
print("\nmodel            MAE     RMSE    R2")
for name, m in models.items():
    pred = m.fit(X_train, y_train).predict(X_test)
    print(f"{name:<16} {mean_absolute_error(y_test, pred):<7.2f} "
          f"{np.sqrt(mean_squared_error(y_test, pred)):<7.2f} {r2_score(y_test, pred):.3f}")
