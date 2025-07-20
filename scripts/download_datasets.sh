#!/bin/bash

# Dataset download script for FedDistill-Eye
# This script provides guidance for downloading the required datasets

echo "======================================"
echo "FedDistill-Eye Dataset Download Guide"
echo "======================================"

echo ""
echo "Please download the following datasets manually:"
echo ""

echo "1. RETFound Benchmark Datasets:"
echo "   URL: https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md"
echo "   Datasets needed:"
echo "   - JSIEC (Primary taxonomy)"
echo "   - APTOS2019 (Diabetic Retinopathy)" 
echo "   - MESSIDOR2 (Diabetic Retinopathy)"
echo "   - IDRiD (Diabetic Retinopathy)"
echo "   - OCTID (OCT Images)"
echo "   - PAPILA (Glaucoma)"
echo "   - Glaucoma_fundus (Glaucoma)"
echo "   - Retina (Multi-disease)"
echo ""

echo "2. Pretrained Self-Supervised Model:"
echo "   URL: https://github.com/abdkhanstd/ATLASS"
echo "   File: best_model.pth (place in root directory)"
echo ""

echo "3. Paper's Pretrained Weights:"
echo "   URL: https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/EsMao40xHZRPjDaK_KjR1A0BbTq9_8bTVs8LIM642cGeyw?e=E6QDtQ"
echo ""

echo "After downloading, organize datasets as follows:"
echo ""
echo "datasets/"
echo "├── JSIEC/"
echo "│   ├── train/ (class folders with images)"
echo "│   ├── val/"
echo "│   └── test/"
echo "├── APTOS2019/"
echo "│   ├── train/"
echo "│   ├── val/"
echo "│   └── test/"
echo "└── (other datasets...)"
echo ""

echo "Each class folder should contain the actual image files (.jpg, .png, etc.)"
echo ""
echo "For detailed dataset organization, see README.md"
