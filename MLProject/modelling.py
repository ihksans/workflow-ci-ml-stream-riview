#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
modelling.py
Diadaptasi untuk MLflow Project di folder Workflow-CI/MLProject.
Melatih model machine learning Scikit-Learn (Logistic Regression) menggunakan TF-IDF
dan MLflow Tracking UI secara lokal dengan fitur autolog aktif serta menerima argument dari luar.
Menyimpan model ke direktori lokal agar dapat dibangun menjadi Docker Image dengan mudah.
"""

import os
import argparse
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn

def main():
    # 1. Parsing Argument Command-Line
    parser = argparse.ArgumentParser(description="Retrain Logistic Regression Sentiment Model using MLflow Project")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength parameter")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations for solver")
    args = parser.parse_args()
    
    print("=== Memulai Eksperimen Modelling (Workflow CI - MLProject) ===")
    print(f"Menggunakan Parameter: C={args.C}, max_iter={args.max_iter}")
    
    # 2. Tentukan Path Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'namadataset_preprocessing', 'steam_reviews_clean.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset tidak ditemukan di {dataset_path}!")
        return
        
    print(f"Membaca dataset dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Tangani nilai NaN jika ada
    df['clean_review'] = df['clean_review'].fillna("")
    
    # 3. Split Dataset
    X = df['clean_review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Ukuran Data Latih: {X_train.shape[0]}")
    print(f"Ukuran Data Uji: {X_test.shape[0]}")
    
    # 4. Vectorization menggunakan TF-IDF
    print("Mengonversi teks ke representasi TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Konfigurasi MLflow
    # Set nama eksperimen (hanya jika dijalankan langsung, bukan lewat mlflow run)
    if "MLFLOW_RUN_ID" not in os.environ:
        experiment_name = "Steam_Sentiment_Retrain_Workflow_CI"
        mlflow.set_experiment(experiment_name)
    
    # Aktifkan Autolog Scikit-Learn
    print("Mengaktifkan MLflow Autolog...")
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
    
    # 6. Inisiasi Model Logistic Regression (dengan parameter masukan dari MLProject)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=42)
    
    # 7. Jalankan Pelatihan Model dalam MLflow Run
    print("Melatih model Scikit-Learn dengan MLflow tracking...")
    with mlflow.start_run(run_name=f"Workflow_CI_LR_C_{args.C}") as run:
        # Melatih model (Autolog secara otomatis merekam parameter, metrik latih, dan model)
        model.fit(X_train_vec, y_train)
        
        # Evaluasi secara manual di console
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print("\nHasil Evaluasi Model:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        
        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, y_pred))
        
        # Informasi Run ID
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print("Model dan metrik berhasil direkam secara otomatis oleh Autolog!")
        
        # Menyimpan model secara lokal ke folder terpisah agar dapat diakses mlflow build-docker
        model_output_path = os.path.join(base_dir, 'model_output')
        if os.path.exists(model_output_path):
            shutil.rmtree(model_output_path)
            
        print(f"Menyimpan model ke folder lokal: {model_output_path}...")
        mlflow.sklearn.save_model(model, model_output_path)
        print("Model berhasil disimpan ke folder lokal!")
        
if __name__ == "__main__":
    main()
