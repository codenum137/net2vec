#!/usr/bin/env python3
"""
收集和汇总不同参数模型的性能指标
生成清晰的性能对比表格
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_model_info(model_name):
    """
    从模型名称中提取模型类型和物理约束参数信息
    例如: kan_hard_0.1 -> (KAN, Hard, 0.1)
         mlp_none -> (MLP, None, 0.0)
    """
    parts = model_name.split('_')
    
    if len(parts) == 2:  # kan_none, mlp_none
        model_type = parts[0].upper()
        physics_type = "None"
        lambda_physics = 0.0
    elif len(parts) == 3:  # kan_hard_0.1, mlp_soft_0.5
        model_type = parts[0].upper()
        physics_type = parts[1].capitalize()  # Hard, Soft
        lambda_physics = float(parts[2])
    else:
        return None, None, None
    
    return model_type, physics_type, lambda_physics

def clean_numeric_data(value):
    """
    清理数值数据，处理百分比和数值混合的情况
    """
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # 处理百分比
    if value_str.endswith('%'):
        return float(value_str[:-1].replace('+', '').replace('-', ''))
    
    # 处理普通数值
    try:
        return float(value_str.replace('+', ''))
    except:
        return np.nan

def collect_performance_data(base_dir):
    """
    收集所有模型的性能数据
    """
    base_path = Path(base_dir)
    all_data = []
    
    # 查找所有numerical_performance_analysis.csv文件
    csv_files = list(base_path.glob("*/numerical/numerical_performance_analysis.csv"))
    
    print(f"找到 {len(csv_files)} 个性能分析文件")
    
    for csv_file in csv_files:
        # 从路径中提取模型名称
        model_name = csv_file.parent.parent.name
        
        # 解析模型信息
        model_type, physics_type, lambda_physics = extract_model_info(model_name)
        
        if model_type is None:
            print(f"⚠️  跳过无法解析的模型: {model_name}")
            continue
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 为每一行添加模型信息
            df['Model_Type'] = model_type
            df['Physics_Type'] = physics_type
            df['Lambda_Physics'] = lambda_physics
            df['Model_Name'] = model_name
            
            all_data.append(df)
            print(f"✅ 成功加载: {model_name}")
            
        except Exception as e:
            print(f"❌ 加载失败 {model_name}: {e}")
    
    if not all_data:
        raise ValueError("没有成功加载任何数据文件")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

def create_performance_summary(df):
    """
    创建性能汇总表格
    """
    # 重新排列列的顺序，把模型信息放在前面
    columns_order = ['Model_Type', 'Physics_Type', 'Lambda_Physics', 'Model_Name', 
                    'Target', 'Metric', 'NSFNet (Training Topology)', 
                    'GBN (Test Topology)', 'Degradation']
    
    df_reordered = df[columns_order]
    
    return df_reordered

def create_pivot_tables(df):
    """
    创建透视表，便于对比分析
    """
    # 清理数据
    df_clean = df.copy()
    
    # 清理数值列
    numeric_columns = ['NSFNet (Training Topology)', 'GBN (Test Topology)']
    for col in numeric_columns:
        df_clean[f'{col}_numeric'] = df_clean[col].apply(clean_numeric_data)
    
    # 清理退化数据
    df_clean['Degradation_Numeric'] = df_clean['Degradation'].apply(clean_numeric_data)
    
    pivot_tables = {}
    
    # 1. 按模型类型和指标的训练集性能透视表（保留原始字符串格式）
    pivot_train = df_clean.pivot_table(
        index=['Model_Type', 'Physics_Type', 'Lambda_Physics'],
        columns=['Target', 'Metric'],
        values='NSFNet (Training Topology)',
        aggfunc='first'
    )
    pivot_tables['Training_Performance'] = pivot_train
    
    # 2. 按模型类型和指标的测试集性能透视表（保留原始字符串格式）
    pivot_test = df_clean.pivot_table(
        index=['Model_Type', 'Physics_Type', 'Lambda_Physics'],
        columns=['Target', 'Metric'],
        values='GBN (Test Topology)',
        aggfunc='first'
    )
    pivot_tables['Test_Performance'] = pivot_test
    
    # 3. 按模型类型和指标的性能退化透视表（数值格式）
    pivot_degradation = df_clean.pivot_table(
        index=['Model_Type', 'Physics_Type', 'Lambda_Physics'],
        columns=['Target', 'Metric'],
        values='Degradation_Numeric',
        aggfunc='first'
    )
    pivot_tables['Performance_Degradation'] = pivot_degradation
    
    # 4. 训练集性能透视表（数值格式，用于排序）
    pivot_train_numeric = df_clean.pivot_table(
        index=['Model_Type', 'Physics_Type', 'Lambda_Physics'],
        columns=['Target', 'Metric'],
        values='NSFNet (Training Topology)_numeric',
        aggfunc='first'
    )
    pivot_tables['Training_Performance_Numeric'] = pivot_train_numeric
    
    # 5. 测试集性能透视表（数值格式，用于排序）
    pivot_test_numeric = df_clean.pivot_table(
        index=['Model_Type', 'Physics_Type', 'Lambda_Physics'],
        columns=['Target', 'Metric'],
        values='GBN (Test Topology)_numeric',
        aggfunc='first'
    )
    pivot_tables['Test_Performance_Numeric'] = pivot_test_numeric
    
    return pivot_tables

def generate_performance_ranking(df):
    """
    生成性能排名表
    """
    rankings = {}
    
    # 清理数据
    df_clean = df.copy()
    
    # 清理数值列
    numeric_columns = ['NSFNet (Training Topology)', 'GBN (Test Topology)']
    for col in numeric_columns:
        df_clean[f'{col}_numeric'] = df_clean[col].apply(clean_numeric_data)
    
    # 清理退化数据
    df_clean['Degradation_Numeric'] = df_clean['Degradation'].apply(clean_numeric_data)
    
    # 按不同指标排名（越小越好的指标）
    better_lower_metrics = ['MAE', 'RMSE', 'MAPE']
    # 按不同指标排名（越大越好的指标）
    better_higher_metrics = ['R2']
    # 特殊指标（越小越好）
    special_metrics = ['NLL']
    
    for target in df_clean['Target'].unique():
        for metric in df_clean['Metric'].unique():
            subset = df_clean[(df_clean['Target'] == target) & (df_clean['Metric'] == metric)]
            
            if subset.empty:
                continue
                
            key = f"{target}_{metric}"
            
            # 确保有有效数据
            train_col = 'NSFNet (Training Topology)_numeric'
            test_col = 'GBN (Test Topology)_numeric'
            
            valid_train = subset.dropna(subset=[train_col])
            valid_test = subset.dropna(subset=[test_col])
            
            if valid_train.empty or valid_test.empty:
                print(f"⚠️  跳过 {key}: 没有有效数值数据")
                continue
            
            if metric in better_lower_metrics or metric in special_metrics:
                # 训练集性能排名（越小越好）
                train_ranking = valid_train.nsmallest(len(valid_train), train_col)[
                    ['Model_Name', 'Model_Type', 'Physics_Type', 'Lambda_Physics', 'NSFNet (Training Topology)', train_col]
                ].reset_index(drop=True)
                train_ranking.index = train_ranking.index + 1
                
                # 测试集性能排名（越小越好）
                test_ranking = valid_test.nsmallest(len(valid_test), test_col)[
                    ['Model_Name', 'Model_Type', 'Physics_Type', 'Lambda_Physics', 'GBN (Test Topology)', test_col]
                ].reset_index(drop=True)
                test_ranking.index = test_ranking.index + 1
                
            elif metric in better_higher_metrics:
                # R2: 越大越好
                train_ranking = valid_train.nlargest(len(valid_train), train_col)[
                    ['Model_Name', 'Model_Type', 'Physics_Type', 'Lambda_Physics', 'NSFNet (Training Topology)', train_col]
                ].reset_index(drop=True)
                train_ranking.index = train_ranking.index + 1
                
                test_ranking = valid_test.nlargest(len(valid_test), test_col)[
                    ['Model_Name', 'Model_Type', 'Physics_Type', 'Lambda_Physics', 'GBN (Test Topology)', test_col]
                ].reset_index(drop=True)
                test_ranking.index = test_ranking.index + 1
            
            rankings[f"{key}_Training"] = train_ranking
            rankings[f"{key}_Test"] = test_ranking
    
    return rankings

def save_results(df, pivot_tables, rankings, output_dir):
    """
    保存所有结果到文件
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. 保存完整的汇总数据
    summary_file = output_path / "performance_summary_complete.csv"
    df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"✅ 完整汇总数据已保存: {summary_file}")
    
    # 2. 保存透视表
    for name, pivot_df in pivot_tables.items():
        pivot_file = output_path / f"pivot_{name.lower()}.csv"
        pivot_df.to_csv(pivot_file, encoding='utf-8')
        print(f"✅ 透视表已保存: {pivot_file}")
    
    # 3. 保存排名表
    ranking_dir = output_path / "rankings"
    ranking_dir.mkdir(exist_ok=True)
    
    for name, ranking_df in rankings.items():
        ranking_file = ranking_dir / f"ranking_{name.lower()}.csv"
        ranking_df.to_csv(ranking_file, encoding='utf-8')
        print(f"✅ 排名表已保存: {ranking_file}")
    
    # 4. 生成可读性强的Markdown报告
    generate_markdown_report(df, pivot_tables, rankings, output_path)

def generate_markdown_report(df, pivot_tables, rankings, output_path):
    """
    生成Markdown格式的可读性报告
    """
    report_file = output_path / "performance_analysis_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 模型性能分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 概览
        f.write("## 1. 实验概览\n\n")
        total_models = df['Model_Name'].nunique()
        model_types = df['Model_Type'].unique()
        f.write(f"- 总模型数量: {total_models}\n")
        f.write(f"- 模型类型: {', '.join(model_types)}\n")
        f.write(f"- 评估指标: DELAY (MAE, RMSE, MAPE, R2, NLL), JITTER (MAE, RMSE, MAPE, R2)\n\n")
        
        # 2. 模型列表
        f.write("## 2. 实验模型列表\n\n")
        model_info = df.groupby(['Model_Type', 'Physics_Type', 'Lambda_Physics']).first()['Model_Name'].reset_index()
        f.write("| 模型类型 | 物理约束类型 | λ_physics | 模型名称 |\n")
        f.write("|---------|-------------|-----------|----------|\n")
        for _, row in model_info.iterrows():
            f.write(f"| {row['Model_Type']} | {row['Physics_Type']} | {row['Lambda_Physics']} | {row['Model_Name']} |\n")
        f.write("\n")
        
        # 3. 关键发现
        f.write("## 3. 关键发现\n\n")
        
        # 找到最佳性能的模型
        df_temp = df.copy()
        df_temp['GBN_numeric'] = df_temp['GBN (Test Topology)'].apply(clean_numeric_data)
        delay_mae_subset = df_temp[(df_temp['Target'] == 'DELAY') & (df_temp['Metric'] == 'MAE')].dropna(subset=['GBN_numeric'])
        
        if not delay_mae_subset.empty:
            delay_mae_best = delay_mae_subset.nsmallest(1, 'GBN_numeric')
            best_model = delay_mae_best.iloc[0]
            f.write(f"### 🏆 DELAY MAE 最佳模型\n")
            f.write(f"- **{best_model['Model_Name']}** (训练集: {best_model['NSFNet (Training Topology)']}, 测试集: {best_model['GBN (Test Topology)']})\n\n")
        
        # 找到泛化性能最好的模型（退化最小）
        df_clean = df.copy()
        df_clean['Degradation_Numeric'] = df_clean['Degradation'].apply(clean_numeric_data)
        delay_mae_generalization_subset = df_clean[(df_clean['Target'] == 'DELAY') & (df_clean['Metric'] == 'MAE')].dropna(subset=['Degradation_Numeric'])
        
        if not delay_mae_generalization_subset.empty:
            delay_mae_generalization = delay_mae_generalization_subset.nsmallest(1, 'Degradation_Numeric')
            best_gen_model = delay_mae_generalization.iloc[0]
            f.write(f"### 🎯 DELAY MAE 泛化性能最佳\n")
            f.write(f"- **{best_gen_model['Model_Name']}** (性能退化: {best_gen_model['Degradation']})\n\n")
        
        # 4. 详细性能表格
        f.write("## 4. 详细性能对比\n\n")
        f.write("### 4.1 DELAY指标对比\n\n")
        
        # 简化表格输出
        f.write("| 模型名称 | 类型 | MAE(训练) | MAE(测试) | RMSE(训练) | RMSE(测试) | R2(训练) | R2(测试) |\n")
        f.write("|----------|------|-----------|-----------|------------|-----------|----------|----------|\n")
        
        for model_name in sorted(df['Model_Name'].unique()):
            model_delay = df[(df['Model_Name'] == model_name) & (df['Target'] == 'DELAY')]
            if len(model_delay) > 0:
                model_type = model_delay.iloc[0]['Model_Type']
                
                mae_data = model_delay[model_delay['Metric'] == 'MAE']
                rmse_data = model_delay[model_delay['Metric'] == 'RMSE']
                r2_data = model_delay[model_delay['Metric'] == 'R2']
                
                mae_train = mae_data['NSFNet (Training Topology)'].values[0] if len(mae_data) > 0 else '-'
                mae_test = mae_data['GBN (Test Topology)'].values[0] if len(mae_data) > 0 else '-'
                rmse_train = rmse_data['NSFNet (Training Topology)'].values[0] if len(rmse_data) > 0 else '-'
                rmse_test = rmse_data['GBN (Test Topology)'].values[0] if len(rmse_data) > 0 else '-'
                r2_train = r2_data['NSFNet (Training Topology)'].values[0] if len(r2_data) > 0 else '-'
                r2_test = r2_data['GBN (Test Topology)'].values[0] if len(r2_data) > 0 else '-'
                
                f.write(f"| {model_name} | {model_type} | {mae_train} | {mae_test} | {rmse_train} | {rmse_test} | {r2_train} | {r2_test} |\n")
        
        f.write("\n")
        
        # 5. 性能退化分析
        f.write("### 4.2 性能退化分析（DELAY MAE）\n\n")
        f.write("| 排名 | 模型名称 | 类型 | 训练集MAE | 测试集MAE | 性能退化 |\n")
        f.write("|------|----------|------|-----------|-----------|----------|\n")
        
        # 按性能退化排序
        delay_mae_degradation = df_clean[(df_clean['Target'] == 'DELAY') & (df_clean['Metric'] == 'MAE')].dropna(subset=['Degradation_Numeric'])
        delay_mae_degradation_sorted = delay_mae_degradation.nsmallest(len(delay_mae_degradation), 'Degradation_Numeric')
        
        for i, (_, row) in enumerate(delay_mae_degradation_sorted.iterrows(), 1):
            f.write(f"| {i} | {row['Model_Name']} | {row['Model_Type']} | {row['NSFNet (Training Topology)']} | {row['GBN (Test Topology)']} | {row['Degradation']} |\n")
        
        f.write("\n")
        
        # 6. 结论和建议
        f.write("## 5. 结论和建议\n\n")
        f.write("### 模型类型对比\n")
        
        # KAN vs MLP 统计
        kan_models = df_clean[df_clean['Model_Type'] == 'KAN']
        mlp_models = df_clean[df_clean['Model_Type'] == 'MLP']
        
        if not kan_models.empty and not mlp_models.empty:
            kan_delay_mae = kan_models[(kan_models['Target'] == 'DELAY') & (kan_models['Metric'] == 'MAE')]['Degradation_Numeric']
            mlp_delay_mae = mlp_models[(mlp_models['Target'] == 'DELAY') & (mlp_models['Metric'] == 'MAE')]['Degradation_Numeric']
            
            if not kan_delay_mae.empty and not mlp_delay_mae.empty:
                kan_avg = kan_delay_mae.mean()
                mlp_avg = mlp_delay_mae.mean()
                f.write(f"- **KAN vs MLP**: KAN平均性能退化 {kan_avg:.2f}%, MLP平均性能退化 {mlp_avg:.2f}%\n")
        
        f.write("- **物理约束策略**: Hard vs Soft vs None的效果对比\n")
        f.write("- **λ_physics参数**: 不同物理约束强度(0.1, 0.5, 1.0)的影响分析\n\n")
        
        f.write("### 推荐配置\n")
        if not delay_mae_generalization_subset.empty:
            best_config = delay_mae_generalization_subset.nsmallest(3, 'Degradation_Numeric')
            f.write("**基于泛化性能的前三名推荐:**\n")
            for i, (_, row) in enumerate(best_config.iterrows(), 1):
                f.write(f"{i}. **{row['Model_Name']}** - 性能退化: {row['Degradation']}\n")
        f.write("\n")
    
    print(f"✅ Markdown报告已保存: {report_file}")

def main():
    """主函数"""
    base_dir = "/home/ubantu/net2vec/experiment_results/opt"
    output_dir = "/home/ubantu/net2vec/experiment_results/opt/summary"
    
    try:
        print("🚀 开始收集模型性能数据...")
        
        # 收集数据
        df = collect_performance_data(base_dir)
        print(f"✅ 成功收集 {len(df)} 条记录，覆盖 {df['Model_Name'].nunique()} 个模型")
        
        # 创建汇总表格
        print("📊 创建性能汇总表格...")
        df_summary = create_performance_summary(df)
        
        # 创建透视表
        print("📈 创建透视表...")
        pivot_tables = create_pivot_tables(df_summary)
        
        # 生成排名
        print("🏆 生成性能排名...")
        rankings = generate_performance_ranking(df_summary)
        
        # 保存结果
        print("💾 保存结果文件...")
        save_results(df_summary, pivot_tables, rankings, output_dir)
        
        print(f"\n🎉 性能分析完成！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"📋 主要文件:")
        print(f"   - performance_summary_complete.csv: 完整汇总数据")
        print(f"   - pivot_*.csv: 透视表")
        print(f"   - rankings/: 性能排名")
        print(f"   - performance_analysis_report.md: 可读性报告")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
