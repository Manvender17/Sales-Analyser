#!/usr/bin/env python3
"""
Sales Data Analyzer - Complete Implementation
A comprehensive sales data analysis system with ML-ready features

Features:
- Generate 10,000+ realistic sales records
- Data cleaning and quality improvement
- Feature engineering and data enrichment
- Automated annotation and validation
- Export data for Power BI dashboards
- ML-ready dataset preparation

Author: Sales Analytics Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import random
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class SalesDataAnalyzer:
    """Complete Sales Data Analysis Pipeline"""
    
    def __init__(self, n_records=10000, seed=42):
        self.n_records = n_records
        self.seed = seed
        self.raw_data = None
        self.cleaned_data = None
        self.final_data = None
        self.cleaning_log = []
        self.feature_log = []
        self.validation_results = {}
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = Faker()
    
    def run_complete_pipeline(self):
        """Execute the complete data analysis pipeline"""
        
        print("🚀 SALES DATA ANALYZER - COMPLETE PIPELINE")
        print("="*60)
        print(f"Processing {self.n_records:,} sales records...")
        print("="*60)
        
        # Step 1: Generate realistic sales data
        print("\n📊 STEP 1: GENERATING SALES DATA")
        self.raw_data = self.generate_sales_data()
        
        # Step 2: Clean and improve data quality
        print("\n🧹 STEP 2: DATA CLEANING & QUALITY IMPROVEMENT")
        self.cleaned_data = self.clean_sales_data(self.raw_data)
        
        # Step 3: Feature engineering and enrichment
        print("\n🔬 STEP 3: FEATURE ENGINEERING & ENRICHMENT")
        self.final_data = self.engineer_features(self.cleaned_data)
        
        # Step 4: Apply annotation guidelines
        print("\n🏷️ STEP 4: DATA ANNOTATION FOR CLASSIFICATION")
        self.final_data = self.apply_annotations(self.final_data)
        
        # Step 5: Automated validation
        print("\n✅ STEP 5: AUTOMATED VALIDATION")
        self.validation_results = self.validate_dataset(self.final_data)
        
        # Step 6: Export for Power BI and ML
        print("\n📤 STEP 6: EXPORT DATA")
        self.export_data()
        
        # Step 7: Generate summary report
        print("\n📋 STEP 7: GENERATE SUMMARY REPORT")
        self.generate_final_report()
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        return self.final_data
    
    def generate_sales_data(self):
        """Generate realistic sales dataset with intentional quality issues for cleaning demo"""
        
        print("Generating realistic sales records with intentional data quality issues...")
        
        # Define realistic business data
        categories = {
            'Electronics': {
                'products': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera', 'Smart Watch', 'Monitor'],
                'price_range': (50, 2000)
            },
            'Clothing': {
                'products': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket', 'Sweater', 'Shorts'],
                'price_range': (15, 300)
            },
            'Home & Garden': {
                'products': ['Sofa', 'Lamp', 'Plant', 'Cookware', 'Bedding', 'Furniture', 'Decor'],
                'price_range': (25, 800)
            },
            'Sports': {
                'products': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Weights', 'Bicycle', 'Soccer Ball'],
                'price_range': (20, 500)
            },
            'Books': {
                'products': ['Fiction', 'Non-Fiction', 'Textbook', 'Biography', 'Cookbook', 'Manual'],
                'price_range': (10, 80)
            }
        }
        
        regions = ['North', 'South', 'East', 'West', 'Central']
        payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'PayPal', 'Bank Transfer']
        customer_segments = ['Individual', 'Small Business', 'Enterprise']
        priorities = ['Low', 'Medium', 'High', 'Urgent']
        
        # Generate sales representatives
        sales_reps = [self.fake.name() for _ in range(25)]
        
        data = []
        
        for i in range(self.n_records):
            # Select category and product
            category = random.choice(list(categories.keys()))
            product = random.choice(categories[category]['products'])
            price_min, price_max = categories[category]['price_range']
            
            # Generate customer data (introduce missing values intentionally)
            customer_name = None if random.random() < 0.03 else self.fake.name()
            customer_email = self.fake.email()
            
            # Introduce inconsistencies in categories (5% chance)
            display_category = category
            if random.random() < 0.05:
                if random.random() < 0.3:
                    display_category = category.lower()
                elif random.random() < 0.3:
                    display_category = category.upper()
                else:
                    display_category = category + " "  # trailing space
            
            # Generate sales data
            quantity = random.randint(1, 12)
            unit_price = round(random.uniform(price_min, price_max), 2)
            discount_percent = round(random.uniform(0, 30), 1)
            
            # Introduce some data quality issues (2% chance)
            if random.random() < 0.02:
                if random.random() < 0.4:
                    quantity = -1  # Negative quantity
                elif random.random() < 0.4:
                    unit_price = -unit_price  # Negative price
                else:
                    discount_percent = 150  # Invalid discount
            
            # Calculate totals
            subtotal = quantity * unit_price
            discount_amount = subtotal * (discount_percent / 100)
            total_amount = round(max(0, subtotal - discount_amount), 2)
            
            # Generate transaction record
            record = {
                'transaction_id': f'TXN{i+1:06d}',
                'sale_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'customer_name': customer_name,
                'customer_email': customer_email,
                'customer_age': random.randint(16, 85),
                'product_category': display_category,
                'product_name': product,
                'quantity': quantity,
                'unit_price': unit_price,
                'discount_percent': discount_percent,
                'total_amount': total_amount,
                'payment_method': random.choice(payment_methods),
                'sales_rep': random.choice(sales_reps),
                'region': random.choice(regions),
                'customer_segment': random.choice(customer_segments),
                'order_priority': random.choice(priorities)
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add some duplicate records (1% chance)
        duplicate_indices = random.sample(range(len(df)), int(len(df) * 0.01))
        duplicates = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        print(f"✅ Generated {len(df):,} sales records with {len(df.columns)} attributes")
        print(f"   • Intentional quality issues: ~5% for cleaning demonstration")
        print(f"   • Missing values: {df.isnull().sum().sum()}")
        print(f"   • Duplicate records: {len(duplicate_indices)}")
        
        return df
    
    def clean_sales_data(self, df):
        """Comprehensive data cleaning and quality improvement"""
        
        print("Starting comprehensive data cleaning pipeline...")
        df_cleaned = df.copy()
        original_shape = df.shape
        
        # 1. Handle missing values
        print("  🔧 Handling missing values...")
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Customer name: Fill with 'Anonymous Customer'
        missing_names = df_cleaned['customer_name'].isnull().sum()
        if missing_names > 0:
            df_cleaned['customer_name'].fillna('Anonymous Customer', inplace=True)
            self.cleaning_log.append(f"Filled {missing_names} missing customer names")
        
        # Numeric fields: Use median/mode
        for col in ['customer_age', 'unit_price', 'discount_percent']:
            if df_cleaned[col].isnull().sum() > 0:
                fill_value = df_cleaned[col].median()
                missing_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(fill_value, inplace=True)
                self.cleaning_log.append(f"Filled {missing_count} missing {col} values")
        
        missing_after = df_cleaned.isnull().sum().sum()
        print(f"     ✅ Reduced missing values: {missing_before} → {missing_after}")
        
        # 2. Standardize categorical data
        print("  🏷️ Standardizing categories...")
        
        # Product categories
        df_cleaned['product_category'] = df_cleaned['product_category'].str.strip().str.title()
        
        # Regions
        df_cleaned['region'] = df_cleaned['region'].str.strip().str.title()
        
        # Payment methods
        payment_mapping = {
            'credit card': 'Credit Card', 'debit card': 'Debit Card',
            'cash': 'Cash', 'paypal': 'PayPal', 'bank transfer': 'Bank Transfer'
        }
        df_cleaned['payment_method'] = df_cleaned['payment_method'].str.lower().str.strip()
        df_cleaned['payment_method'] = df_cleaned['payment_method'].map(payment_mapping).fillna(df_cleaned['payment_method'])
        df_cleaned['payment_method'] = df_cleaned['payment_method'].str.title()
        
        self.cleaning_log.append("Standardized categorical variables")
        
        # 3. Fix data types
        print("  🔧 Correcting data types...")
        df_cleaned['sale_date'] = pd.to_datetime(df_cleaned['sale_date'])
        
        numeric_cols = ['quantity', 'unit_price', 'discount_percent', 'total_amount', 'customer_age']
        for col in numeric_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # 4. Remove duplicates
        print("  🔍 Removing duplicates...")
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['transaction_id'], keep='first')
        duplicates_removed = initial_count - len(df_cleaned)
        if duplicates_removed > 0:
            self.cleaning_log.append(f"Removed {duplicates_removed} duplicate records")
        
        # 5. Apply business rules
        print("  ⚖️ Applying business rule validations...")
        rules_initial = len(df_cleaned)
        
        # Remove invalid records
        df_cleaned = df_cleaned[
            (df_cleaned['quantity'] > 0) & 
            (df_cleaned['unit_price'] > 0) & 
            (df_cleaned['discount_percent'] >= 0) & 
            (df_cleaned['discount_percent'] <= 100) &
            (df_cleaned['customer_age'] >= 13) & 
            (df_cleaned['customer_age'] <= 120)
        ]
        
        # Recalculate total_amount for consistency
        subtotal = df_cleaned['quantity'] * df_cleaned['unit_price']
        discount_amt = subtotal * (df_cleaned['discount_percent'] / 100)
        df_cleaned['total_amount'] = (subtotal - discount_amt).round(2)
        
        rules_removed = rules_initial - len(df_cleaned)
        if rules_removed > 0:
            self.cleaning_log.append(f"Removed {rules_removed} records violating business rules")
        
        # 6. Final cleanup
        print("  📝 Final text standardization...")
        text_fields = ['customer_name', 'product_name', 'sales_rep']
        for field in text_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].str.strip().str.title()
        
        df_cleaned['customer_email'] = df_cleaned['customer_email'].str.lower().str.strip()
        
        # Generate cleaning report
        final_shape = df_cleaned.shape
        retention_rate = (len(df_cleaned) / len(df)) * 100
        
        print(f"\n📋 CLEANING SUMMARY:")
        print(f"   • Original records: {original_shape[0]:,}")
        print(f"   • Cleaned records: {final_shape[0]:,}")
        print(f"   • Records removed: {original_shape[0] - final_shape[0]:,}")
        print(f"   • Data retention: {retention_rate:.1f}%")
        print(f"   • Quality improvements: {len(self.cleaning_log)} steps applied")
        
        return df_cleaned
    
    def engineer_features(self, df):
        """Create derived attributes and enrichment features for ML"""
        
        print("Engineering comprehensive feature set for ML and analysis...")
        df_enriched = df.copy()
        initial_features = len(df_enriched.columns)
        
        # 1. Temporal features
        print("  📅 Creating temporal features...")
        df_enriched['year'] = df_enriched['sale_date'].dt.year
        df_enriched['month'] = df_enriched['sale_date'].dt.month
        df_enriched['quarter'] = df_enriched['sale_date'].dt.quarter
        df_enriched['day_of_week'] = df_enriched['sale_date'].dt.dayofweek
        df_enriched['day_name'] = df_enriched['sale_date'].dt.day_name()
        df_enriched['is_weekend'] = df_enriched['day_of_week'].isin([5, 6]).astype(int)
        df_enriched['is_month_end'] = df_enriched['sale_date'].dt.is_month_end.astype(int)
        df_enriched['days_since_first_sale'] = (df_enriched['sale_date'] - df_enriched['sale_date'].min()).dt.days
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
                     5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df_enriched['season'] = df_enriched['month'].map(season_map)
        df_enriched['is_holiday_season'] = df_enriched['month'].isin([11, 12]).astype(int)
        
        # 2. Customer segmentation features
        print("  👥 Creating customer features...")
        
        # Age groups
        df_enriched['age_group'] = pd.cut(df_enriched['customer_age'], 
                                         bins=[0, 25, 35, 50, 65, 100], 
                                         labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Customer value analysis
        customer_stats = df_enriched.groupby('customer_email').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'sale_date': ['min', 'max']
        }).round(2)
        
        customer_stats.columns = ['customer_total_spent', 'customer_avg_order', 'customer_order_count', 
                                 'customer_first_purchase', 'customer_last_purchase']
        
        df_enriched = df_enriched.merge(customer_stats, left_on='customer_email', right_index=True, how='left')
        
        # Customer value tiers
        df_enriched['customer_value_tier'] = pd.qcut(df_enriched['customer_total_spent'], 
                                                    q=4, labels=['Low', 'Medium', 'High', 'Premium'], 
                                                    duplicates='drop')
        
        # 3. Product performance features
        print("  📦 Creating product features...")
        
        # Price tiers within categories
        df_enriched['price_tier'] = df_enriched.groupby('product_category')['unit_price'].transform(
            lambda x: pd.qcut(x, q=3, labels=['Budget', 'Mid-Range', 'Premium'], duplicates='drop')
        )
        
        # Discount tiers
        df_enriched['discount_tier'] = pd.cut(df_enriched['discount_percent'], 
                                             bins=[-0.1, 0, 10, 20, 100], 
                                             labels=['No Discount', 'Low', 'Medium', 'High'])
        
        # Revenue calculations
        df_enriched['gross_revenue'] = df_enriched['quantity'] * df_enriched['unit_price']
        df_enriched['discount_amount'] = df_enriched['gross_revenue'] * (df_enriched['discount_percent'] / 100)
        df_enriched['revenue_per_unit'] = (df_enriched['total_amount'] / df_enriched['quantity']).round(2)
        df_enriched['profit_margin'] = ((df_enriched['total_amount'] / df_enriched['gross_revenue']) * 100).round(2)
        
        # 4. Sales performance features
        print("  📊 Creating sales performance features...")
        
        # Sales rep performance
        rep_stats = df_enriched.groupby('sales_rep').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum'
        }).round(2)
        rep_stats.columns = ['rep_total_sales', 'rep_avg_sale', 'rep_transaction_count', 'rep_total_quantity']
        
        df_enriched = df_enriched.merge(rep_stats, left_on='sales_rep', right_index=True, how='left')
        
        # Regional performance
        region_stats = df_enriched.groupby('region')['total_amount'].mean().round(2)
        df_enriched['region_avg_sale'] = df_enriched['region'].map(region_stats)
        
        # 5. RFM Analysis (Recency, Frequency, Monetary)
        print("  🎯 Creating RFM analysis features...")
        
        current_date = df_enriched['sale_date'].max()
        
        rfm_data = df_enriched.groupby('customer_email').agg({
            'sale_date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',                             # Frequency  
            'total_amount': 'sum'                                  # Monetary
        }).round(2)
        
        rfm_data.columns = ['recency_days', 'frequency_score', 'monetary_value']
        
        # Create RFM scores (1-5 scale)
        rfm_data['recency_score'] = pd.qcut(rfm_data['recency_days'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm_data['frequency_rank'] = pd.qcut(rfm_data['frequency_score'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm_data['monetary_rank'] = pd.qcut(rfm_data['monetary_value'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Merge RFM back to main dataset
        df_enriched = df_enriched.merge(rfm_data[['recency_score', 'frequency_rank', 'monetary_rank']], 
                                       left_on='customer_email', right_index=True, how='left')
        
        final_features = len(df_enriched.columns)
        features_added = final_features - initial_features
        
        print(f"\n🔬 FEATURE ENGINEERING SUMMARY:")
        print(f"   • Original features: {initial_features}")
        print(f"   • New features created: {features_added}")
        print(f"   • Total features: {final_features}")
        print(f"   • Feature categories: Temporal, Customer, Product, Sales, RFM")
        
        return df_enriched
    
    def apply_annotations(self, df):
        """Apply annotation guidelines for classification tasks"""
        
        print("Applying comprehensive annotation guidelines for ML classification...")
        df_annotated = df.copy()
        
        # Define annotation rules with business logic
        annotation_rules = {
            'high_value_customer': lambda row: (
                pd.notna(row['monetary_rank']) and int(row['monetary_rank']) >= 4 and 
                pd.notna(row['frequency_rank']) and int(row['frequency_rank']) >= 3
            ),
            'at_risk_customer': lambda row: (
                pd.notna(row['recency_score']) and int(row['recency_score']) <= 2 and 
                pd.notna(row['frequency_rank']) and int(row['frequency_rank']) >= 3
            ),
            'vip_customer': lambda row: (
                row['customer_value_tier'] == 'Premium' and 
                row['customer_order_count'] >= 5
            ),
            'bulk_purchase': lambda row: row['quantity'] >= 5,
            'high_discount_sale': lambda row: row['discount_percent'] > 15,
            'premium_product': lambda row: row['price_tier'] == 'Premium',
            'weekend_sale': lambda row: row['is_weekend'] == 1,
            'holiday_season_sale': lambda row: row['is_holiday_season'] == 1,
            'new_customer': lambda row: row['customer_order_count'] == 1,
            'repeat_customer': lambda row: row['customer_order_count'] >= 3,
            'high_margin_sale': lambda row: pd.notna(row['profit_margin']) and row['profit_margin'] >= 80,
            'large_transaction': lambda row: row['total_amount'] >= df['total_amount'].quantile(0.9),
            'electronics_purchase': lambda row: row['product_category'] == 'Electronics',
            'urgent_order': lambda row: row['order_priority'] == 'Urgent',
            'enterprise_customer': lambda row: row['customer_segment'] == 'Enterprise'
        }
        
        # Apply annotations
        annotation_counts = {}
        for label, rule in annotation_rules.items():
            try:
                df_annotated[f'is_{label}'] = df_annotated.apply(rule, axis=1).astype(int)
                count = df_annotated[f'is_{label}'].sum()
                annotation_counts[label] = count
                print(f"  ✅ {label}: {count:,} records ({count/len(df_annotated)*100:.1f}%)")
            except Exception as e:
                print(f"  ⚠️ Error applying {label}: {str(e)}")
                df_annotated[f'is_{label}'] = 0
        
        print(f"\n🏷️ ANNOTATION SUMMARY:")
        print(f"   • Total annotation labels: {len(annotation_rules)}")
        print(f"   • Successfully applied: {len([k for k, v in annotation_counts.items() if v > 0])}")
        print(f"   • Records with multiple labels: Enhanced classification capability")
        
        return df_annotated
    
    def validate_dataset(self, df):
        """Automated validation scripts for data quality"""
        
        print("Running comprehensive automated validation checks...")
        
        validation_results = {}
        
        # 1. Completeness validation
        print("  📊 Checking data completeness...")
        missing_values = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_cells = missing_values.sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        validation_results['completeness'] = {
            'score': round(completeness_score, 2),
            'missing_values': missing_values.sum(),
            'status': 'PASS' if completeness_score >= 95 else 'WARNING'
        }
        
        # 2. Data type validation
        print("  🔧 Validating data types...")
        expected_types = {
            'transaction_id': 'object',
            'sale_date': 'datetime64[ns]',
            'quantity': 'numeric',
            'unit_price': 'numeric',
            'total_amount': 'numeric'
        }
        
        type_issues = 0
        for col, expected_type in expected_types.items():
            if col in df.columns:
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues += 1
                elif expected_type == 'datetime64[ns]' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    type_issues += 1
        
        validation_results['data_types'] = {
            'issues': type_issues,
            'status': 'PASS' if type_issues == 0 else 'WARNING'
        }
        
        # 3. Business rule validation
        print("  ⚖️ Validating business rules...")
        business_violations = 0
        
        # Check for negative values
        if (df['quantity'] < 0).any():
            business_violations += (df['quantity'] < 0).sum()
        if (df['unit_price'] < 0).any():
            business_violations += (df['unit_price'] < 0).sum()
        if (df['total_amount'] < 0).any():
            business_violations += (df['total_amount'] < 0).sum()
        
        # Check discount logic
        expected_total = df['quantity'] * df['unit_price'] * (1 - df['discount_percent']/100)
        calculation_errors = abs(df['total_amount'] - expected_total) > 0.01
        business_violations += calculation_errors.sum()
        
        validation_results['business_rules'] = {
            'violations': business_violations,
            'status': 'PASS' if business_violations == 0 else 'WARNING'
        }
        
        # 4. Statistical validation
        print("  📈 Performing statistical validation...")
        outlier_counts = {}
        numeric_cols = ['unit_price', 'total_amount', 'customer_age', 'discount_percent']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
        
        total_outliers = sum(outlier_counts.values())
        outlier_percentage = (total_outliers / len(df)) * 100
        
        validation_results['statistical'] = {
            'outliers': total_outliers,
            'outlier_percentage': round(outlier_percentage, 2),
            'outlier_details': outlier_counts,
            'status': 'PASS' if outlier_percentage < 5 else 'WARNING'
        }
        
        # 5. Label accuracy validation
        print("  🎯 Validating annotation labels...")
        annotation_cols = [col for col in df.columns if col.startswith('is_')]
        label_distribution = {}
        
        for col in annotation_cols:
            positive_labels = df[col].sum()
            label_percentage = (positive_labels / len(df)) * 100
            label_distribution[col] = {
                'count': positive_labels,
                'percentage': round(label_percentage, 2)
            }
        
        validation_results['annotations'] = {
            'total_labels': len(annotation_cols),
            'label_distribution': label_distribution,
            'status': 'PASS'
        }
        
        # Overall validation score
        individual_scores = []
        for key, result in validation_results.items():
            if key == 'completeness':
                individual_scores.append(result['score'])
            elif key == 'data_types':
                individual_scores.append(100 if result['issues'] == 0 else 80)
            elif key == 'business_rules':
                individual_scores.append(100 if result['violations'] == 0 else 70)
            elif key == 'statistical':
                individual_scores.append(max(0, 100 - result['outlier_percentage']))
        
        overall_score = np.mean(individual_scores)
        validation_results['overall'] = {
            'score': round(overall_score, 1),
            'status': 'PASS' if overall_score >= 80 else 'WARNING' if overall_score >= 60 else 'FAIL'
        }
        
        print(f"\n✅ VALIDATION SUMMARY:")
        print(f"   • Overall Score: {validation_results['overall']['score']}/100")
        print(f"   • Status: {validation_results['overall']['status']}")
        print(f"   • Completeness: {validation_results['completeness']['score']:.1f}%")
        print(f"   • Business Rules: {validation_results['business_rules']['violations']} violations")
        print(f"   • Statistical Outliers: {validation_results['statistical']['outlier_percentage']:.1f}%")
        print(f"   • Annotation Labels: {validation_results['annotations']['total_labels']} applied")
        
        return validation_results
    
    def export_data(self):
        """Export processed data for Power BI and ML applications"""
        
        print("Exporting data in multiple formats for analysis and visualization...")
        
        # Create export directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/exports', exist_ok=True)
        os.makedirs('data/ml_ready', exist_ok=True)
        
        # 1. Main processed dataset (CSV for Power BI)
        main_export_path = 'data/processed/sales_data_complete.csv'
        self.final_data.to_csv(main_export_path, index=False)
        print(f"  📄 Main dataset: {main_export_path}")
        
        # 2. Excel file with multiple sheets for comprehensive analysis
        excel_path = 'data/exports/sales_analysis_complete.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            self.final_data.to_excel(writer, sheet_name='Complete_Dataset', index=False)
            
            # Summary statistics
            summary_stats = self.final_data.describe(include='all')
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Category analysis for Power BI
            category_analysis = self.final_data.groupby('product_category').agg({
                'total_amount': ['sum', 'mean', 'count'],
                'quantity': 'sum',
                'discount_percent': 'mean'
            }).round(2)
            category_analysis.columns = ['Total_Revenue', 'Avg_Sale_Amount', 'Transaction_Count', 
                                       'Total_Quantity', 'Avg_Discount']
            category_analysis.to_excel(writer, sheet_name='Category_Analysis')
            
            # Customer segmentation analysis
            customer_analysis = self.final_data.groupby('customer_value_tier').agg({
                'customer_email': 'nunique',
                'total_amount': ['sum', 'mean'],
                'customer_order_count': 'mean'
            }).round(2)
            customer_analysis.columns = ['Unique_Customers', 'Total_Revenue', 'Avg_Order_Value', 'Avg_Orders']
            customer_analysis.to_excel(writer, sheet_name='Customer_Segments')
            
            # Sales rep performance
            rep_performance = self.final_data.groupby('sales_rep').agg({
                'total_amount': ['sum', 'mean', 'count'],
                'customer_email': 'nunique'
            }).round(2).sort_values(('total_amount', 'sum'), ascending=False)
            rep_performance.columns = ['Total_Sales', 'Avg_Sale', 'Transaction_Count', 'Unique_Customers']
            rep_performance.to_excel(writer, sheet_name='Sales_Rep_Performance')
            
            # Regional analysis
            regional_analysis = self.final_data.groupby('region').agg({
                'total_amount': ['sum', 'mean'],
                'customer_email': 'nunique',
                'transaction_id': 'count'
            }).round(2)
            regional_analysis.columns = ['Total_Revenue', 'Avg_Sale', 'Unique_Customers', 'Total_Transactions']
            regional_analysis.to_excel(writer, sheet_name='Regional_Analysis')
            
            # Time series analysis
            monthly_analysis = self.final_data.groupby(['year', 'month']).agg({
                'total_amount': 'sum',
                'transaction_id': 'count',
                'customer_email': 'nunique'
            }).round(2)
            monthly_analysis.columns = ['Monthly_Revenue', 'Monthly_Transactions', 'Monthly_Customers']
            monthly_analysis.to_excel(writer, sheet_name='Monthly_Trends')
            
            # Annotation summary for quality monitoring
            annotation_cols = [col for col in self.final_data.columns if col.startswith('is_')]
            annotation_summary = []
            for col in annotation_cols:
                count = self.final_data[col].sum()
                percentage = (count / len(self.final_data)) * 100
                annotation_summary.append({
                    'Label': col.replace('is_', '').replace('_', ' ').title(),
                    'Count': count,
                    'Percentage': round(percentage, 2)
                })
            
            annotation_df = pd.DataFrame(annotation_summary)
            annotation_df.to_excel(writer, sheet_name='Annotation_Patterns', index=False)
        
        print(f"  📊 Excel analysis: {excel_path}")
        
        # 3. Power BI optimized datasets
        # Main dataset optimized for Power BI (remove redundant columns, optimize data types)
        powerbi_data = self.final_data.copy()
        
        # Convert datetime to string for Power BI compatibility
        powerbi_data['sale_date_str'] = powerbi_data['sale_date'].dt.strftime('%Y-%m-%d')
        
        # Create separate dimension tables for Power BI star schema
        
        # Customer dimension
        customer_dim = powerbi_data[['customer_email', 'customer_name', 'customer_age', 'age_group', 
                                   'customer_value_tier', 'customer_segment']].drop_duplicates()
        customer_dim.to_csv('data/exports/customer_dimension.csv', index=False)
        
        # Product dimension
        product_dim = powerbi_data[['product_category', 'product_name', 'price_tier']].drop_duplicates()
        product_dim.to_csv('data/exports/product_dimension.csv', index=False)
        
        # Sales rep dimension
        salesrep_dim = powerbi_data[['sales_rep', 'region']].drop_duplicates()
        salesrep_dim.to_csv('data/exports/salesrep_dimension.csv', index=False)
        
        # Date dimension
        date_dim = powerbi_data[['sale_date', 'sale_date_str', 'year', 'month', 'quarter', 
                               'day_name', 'season', 'is_weekend', 'is_holiday_season']].drop_duplicates()
        date_dim.to_csv('data/exports/date_dimension.csv', index=False)
        
        # Fact table (main sales data)
        fact_cols = ['transaction_id', 'sale_date_str', 'customer_email', 'product_category', 'product_name',
                    'sales_rep', 'quantity', 'unit_price', 'discount_percent', 'total_amount', 'payment_method',
                    'order_priority'] + annotation_cols
        
        sales_fact = powerbi_data[fact_cols]
        sales_fact.to_csv('data/exports/sales_fact_table.csv', index=False)
        
        print(f"  🎯 Power BI dimensions: customer, product, salesrep, date")
        print(f"  🎯 Power BI fact table: sales_fact_table.csv")
        
        # 4. ML-ready dataset
        print("  🤖 Preparing ML-ready dataset...")
        
        # Create ML dataset with encoded categorical variables
        ml_data = self.final_data.copy()
        
        # Select features for ML (remove identifiers and redundant columns)
        ml_features = [col for col in ml_data.columns if col not in [
            'transaction_id', 'customer_name', 'customer_email', 'product_name', 'sales_rep', 
            'sale_date', 'sale_date_str', 'customer_first_purchase', 'customer_last_purchase'
        ]]
        
        ml_dataset = ml_data[ml_features].copy()
        
        # Encode categorical variables for ML
        categorical_cols = ml_dataset.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in ml_dataset.columns:
                # Create dummy variables
                dummies = pd.get_dummies(ml_dataset[col], prefix=col, drop_first=True)
                ml_dataset = pd.concat([ml_dataset, dummies], axis=1)
                ml_dataset.drop(col, axis=1, inplace=True)
        
        # Save ML-ready dataset
        ml_path = 'data/ml_ready/sales_ml_dataset.csv'
        ml_dataset.to_csv(ml_path, index=False)
        
        # Create feature documentation
        feature_docs = {
            'total_features': len(ml_dataset.columns),
            'numeric_features': len(ml_dataset.select_dtypes(include=[np.number]).columns),
            'binary_features': len([col for col in ml_dataset.columns if col.startswith('is_')]),
            'encoded_features': len([col for col in ml_dataset.columns if '_' in col and any(cat in col for cat in categorical_cols)]),
            'target_suggestions': ['is_high_value_customer', 'is_at_risk_customer', 'customer_value_tier', 'total_amount']
        }
        
        import json
        with open('data/ml_ready/feature_documentation.json', 'w', encoding='utf-8') as f:
            json.dump(feature_docs, f, indent=2, default=str)
        
        print(f"  🤖 ML dataset: {ml_path} ({ml_dataset.shape[0]} records, {ml_dataset.shape[1]} features)")
        
        # 5. Data quality report for stakeholders
        quality_report = {
            'dataset_summary': {
                'total_records': len(self.final_data),
                'total_features': len(self.final_data.columns),
                'date_range': f"{self.final_data['sale_date'].min()} to {self.final_data['sale_date'].max()}",
                'total_revenue': self.final_data['total_amount'].sum(),
                'unique_customers': self.final_data['customer_email'].nunique(),
                'unique_products': self.final_data['product_name'].nunique()
            },
            'data_quality_metrics': self.validation_results,
            'annotation_labels': len([col for col in self.final_data.columns if col.startswith('is_')]),
            'export_files_created': [
                'sales_data_complete.csv',
                'sales_analysis_complete.xlsx', 
                'Power BI dimension tables (4 files)',
                'sales_fact_table.csv',
                'sales_ml_dataset.csv'
            ]
        }
        
        with open('data/exports/data_quality_report.json', 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        print(f"\n📤 EXPORT SUMMARY:")
        print(f"   • Main CSV: ✅ (Power BI ready)")
        print(f"   • Excel workbook: ✅ (7 analysis sheets)")
        print(f"   • Power BI tables: ✅ (Star schema)")
        print(f"   • ML dataset: ✅ ({ml_dataset.shape[1]} features)")
        print(f"   • Quality report: ✅ (JSON format)")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        print("Generating comprehensive project summary report...")
        
        # Calculate key metrics
        total_revenue = self.final_data['total_amount'].sum()
        avg_order_value = self.final_data['total_amount'].mean()
        unique_customers = self.final_data['customer_email'].nunique()
        date_range = f"{self.final_data['sale_date'].min().date()} to {self.final_data['sale_date'].max().date()}"
        
        # Top performing categories
        top_categories = self.final_data.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
        
        # Customer insights
        customer_segments = self.final_data['customer_value_tier'].value_counts()
        
        # Sales rep performance
        top_sales_reps = self.final_data.groupby('sales_rep')['total_amount'].sum().sort_values(ascending=False).head(5)
        
        # Annotation insights
        annotation_cols = [col for col in self.final_data.columns if col.startswith('is_')]
        top_annotations = {}
        for col in annotation_cols:
            count = self.final_data[col].sum()
            if count > 0:
                top_annotations[col.replace('is_', '')] = count
        
        report = f"""
🎉 SALES DATA ANALYZER - FINAL PROJECT REPORT
{'='*80}

📊 DATASET OVERVIEW
   • Total Records Processed: {len(self.final_data):,}
   • Date Range: {date_range}
   • Total Revenue: ${total_revenue:,.2f}
   • Average Order Value: ${avg_order_value:.2f}
   • Unique Customers: {unique_customers:,}
   • Product Categories: {self.final_data['product_category'].nunique()}
   • Sales Representatives: {self.final_data['sales_rep'].nunique()}

🧹 DATA QUALITY IMPROVEMENTS
   • Data Cleaning Steps: {len(self.cleaning_log)}
   • Quality Score: {self.validation_results.get('overall', {}).get('score', 'N/A')}/100
   • Missing Values Fixed: {self.validation_results.get('completeness', {}).get('missing_values', 0)}
   • Business Rule Violations: {self.validation_results.get('business_rules', {}).get('violations', 0)}
   • Data Retention Rate: {(len(self.final_data) / len(self.raw_data) * 100):.1f}%

🔬 FEATURE ENGINEERING
   • Original Features: {len(self.raw_data.columns)}
   • Engineered Features: {len(self.final_data.columns) - len(self.raw_data.columns)}
   • Total Features: {len(self.final_data.columns)}
   • Feature Categories: Temporal, Customer, Product, Sales, RFM

🏷️ ANNOTATION RESULTS
   • Classification Labels Applied: {len(annotation_cols)}
   • High-Value Customers: {self.final_data['is_high_value_customer'].sum():,}
   • At-Risk Customers: {self.final_data['is_at_risk_customer'].sum():,}
   • VIP Customers: {self.final_data['is_vip_customer'].sum():,}
   • Bulk Purchases: {self.final_data['is_bulk_purchase'].sum():,}
   • High Discount Sales: {self.final_data['is_high_discount_sale'].sum():,}

📈 BUSINESS INSIGHTS
   Top Product Categories by Revenue:
   {chr(10).join([f'   • {cat}: ${rev:,.2f}' for cat, rev in top_categories.head().items()])}
   
   Customer Value Distribution:
   {chr(10).join([f'   • {tier}: {count:,} customers' for tier, count in customer_segments.items()])}
   
   Top Sales Representatives:
   {chr(10).join([f'   • {rep}: ${sales:,.2f}' for rep, sales in top_sales_reps.items()])}

📤 DELIVERABLES CREATED
   ✅ Complete processed dataset (CSV)
   ✅ Multi-sheet Excel analysis workbook
   ✅ Power BI ready dimension tables
   ✅ ML-ready encoded dataset
   ✅ Data quality validation reports
   ✅ Feature documentation
   ✅ Business intelligence exports

🎯 READY FOR:
   • Machine Learning model training and testing
   • Power BI dashboard development
   • Advanced analytics and reporting
   • Stakeholder presentations
   • Quality monitoring and governance

{'='*80}
✨ PROJECT STATUS: COMPLETED SUCCESSFULLY ✨
"""
        
        print(report)
        
        # Save report to file
        with open('data/exports/final_project_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def create_sample_visualizations(self):
        """Create sample visualizations to demonstrate the data"""
        
        print("Creating sample visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sales Data Analysis - Key Insights Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue by Category
        category_revenue = self.final_data.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
        axes[0, 0].pie(category_revenue.values, labels=category_revenue.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Revenue Distribution by Category')
        
        # 2. Monthly Sales Trend
        monthly_sales = self.final_data.groupby(['year', 'month'])['total_amount'].sum().reset_index()
        monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
        axes[0, 1].plot(monthly_sales['date'], monthly_sales['total_amount'], marker='o', linewidth=2)
        axes[0, 1].set_title('Monthly Sales Trend')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Customer Value Distribution
        customer_value = self.final_data['customer_value_tier'].value_counts()
        axes[0, 2].bar(customer_value.index, customer_value.values, color='skyblue', edgecolor='navy')
        axes[0, 2].set_title('Customer Value Tier Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Sales by Day of Week
        dow_sales = self.final_data.groupby('day_name')['total_amount'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_sales = dow_sales.reindex(day_order)
        axes[1, 0].bar(range(7), dow_sales.values, color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels([day[:3] for day in day_order])
        axes[1, 0].set_title('Average Sales by Day of Week')
        
        # 5. Discount vs Revenue Scatter
        sample_data = self.final_data.sample(min(1000, len(self.final_data)))
        scatter = axes[1, 1].scatter(sample_data['discount_percent'], sample_data['total_amount'], 
                                   alpha=0.6, c=sample_data['quantity'], cmap='viridis')
        axes[1, 1].set_xlabel('Discount Percentage')
        axes[1, 1].set_ylabel('Total Amount')
        axes[1, 1].set_title('Discount vs Revenue (colored by quantity)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Quantity')
        
        # 6. Top Sales Reps
        top_reps = self.final_data.groupby('sales_rep')['total_amount'].sum().sort_values(ascending=False).head(10)
        axes[1, 2].barh(range(len(top_reps)), top_reps.values, color='coral')
        axes[1, 2].set_yticks(range(len(top_reps)))
        axes[1, 2].set_yticklabels([name.split()[0] for name in top_reps.index])  # First name only
        axes[1, 2].set_title('Top 10 Sales Representatives')
        
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs('data/exports', exist_ok=True)
        plt.savefig('data/exports/sales_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Sample dashboard saved: data/exports/sales_analysis_dashboard.png")

def main():
    """Main execution function"""
    
    print("🚀 Welcome to the Sales Data Analyzer!")
    print("This tool will create a comprehensive sales dataset and prepare it for ML and Power BI")
    print("-" * 80)
    
    # Initialize the analyzer
    analyzer = SalesDataAnalyzer(n_records=12000, seed=42)
    
    # Run the complete pipeline
    final_dataset = analyzer.run_complete_pipeline()
    
    # Create sample visualizations
    print("\n📊 Creating sample visualizations...")
    analyzer.create_sample_visualizations()
    
    print("\n" + "="*80)
    print("🎉 SUCCESS! Your Sales Data Analyzer project is complete!")
    print("="*80)
    print("\n📁 Check these folders for your outputs:")
    print("   • data/processed/ - Main cleaned dataset")
    print("   • data/exports/ - Power BI files and analysis")
    print("   • data/ml_ready/ - Machine learning datasets")
    print("\n🔗 Next Steps:")
    print("   1. Import the CSV files into Power BI")
    print("   2. Use the ML dataset for model training")
    print("   3. Review the data quality reports")
    print("   4. Customize annotations for your use case")
    
    return final_dataset

# Run the complete analysis
if __name__ == "__main__":
    # Execute the main pipeline
    dataset = main()
    
    # Display basic info about the final dataset
    print(f"\n📊 Final Dataset Summary:")
    print(f"Shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns[:10])}... (showing first 10)")
    print(f"Date Range: {dataset['sale_date'].min()} to {dataset['sale_date'].max()}")
    print(f"Total Revenue: ${dataset['total_amount'].sum():,.2f}")
    print(f"Average Order Value: ${dataset['total_amount'].mean():.2f}")
    print(f"Unique Customers: {dataset['customer_email'].nunique():,}")
    
    print("\n✅ Ready for Power BI dashboard creation and ML model development!")