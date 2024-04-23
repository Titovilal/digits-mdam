from sklearn.metrics import f1_score, precision_score, recall_score

def calcular_metricas(modelo, X_test, y_test):
    # Hacer predicciones con el modelo
    y_pred = modelo.predict(X_test)

    # Calcular las métricas
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    return f1, precision, recall
