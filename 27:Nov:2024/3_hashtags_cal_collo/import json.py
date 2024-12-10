import json
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

def load_and_extract_hashtags(json_data):
    """
    从JSON数据中提取所有topPosts的hashtags
    返回一个列表的列表，每个内部列表包含一个post中的所有hashtags
    """
    # 解析JSON数据
    data = json.loads(json_data)
    
    # 提取所有topPosts中的hashtags
    all_post_hashtags = []
    
    # 遍历所有post
    for post in data:
        # 检查post是否有hashtags字段
        if 'hashtags' in post:
            # 转换所有hashtag为小写以标准化
            hashtags = [tag.lower() for tag in post['hashtags']]
            if hashtags:  # 只添加非空的hashtag列表
                all_post_hashtags.append(hashtags)
    
    return all_post_hashtags

def calculate_cooccurrence_matrix(hashtag_lists):
    """
    计算hashtag的共现矩阵
    返回共现矩阵和所有唯一hashtags的列表
    """
    # 获取所有唯一的hashtags
    unique_hashtags = sorted(list(set([tag for sublist in hashtag_lists for tag in sublist])))
    
    # 创建共现矩阵
    cooccurrence_matrix = pd.DataFrame(0, 
                                     index=unique_hashtags,
                                     columns=unique_hashtags)
    
    # 计算共现次数
    for hashtags in hashtag_lists:
        # 获取当前post中所有hashtag的组合
        for tag1, tag2 in combinations(hashtags, 2):
            cooccurrence_matrix.loc[tag1, tag2] += 1
            cooccurrence_matrix.loc[tag2, tag1] += 1
    
    return cooccurrence_matrix, unique_hashtags

def calculate_pmi(cooccurrence_matrix):
    """
    计算PMI (Pointwise Mutual Information)
    PMI = log(P(x,y)/(P(x)P(y)))
    """
    # 计算每个hashtag的边缘概率
    total_cooccurrences = cooccurrence_matrix.sum().sum() / 2
    marginal_probs = cooccurrence_matrix.sum() / total_cooccurrences
    
    # 计算PMI矩阵
    pmi_matrix = pd.DataFrame(0, 
                            index=cooccurrence_matrix.index,
                            columns=cooccurrence_matrix.columns)
    
    for tag1 in cooccurrence_matrix.index:
        for tag2 in cooccurrence_matrix.columns:
            if tag1 != tag2:
                joint_prob = cooccurrence_matrix.loc[tag1, tag2] / total_cooccurrences
                if joint_prob > 0:  # 避免log(0)
                    pmi = np.log2(joint_prob / (marginal_probs[tag1] * marginal_probs[tag2]))
                    pmi_matrix.loc[tag1, tag2] = pmi
    
    return pmi_matrix

def get_top_collocations(pmi_matrix, n=10):
    """
    获取PMI值最高的n个collocation
    """
    # 创建一个Series存储所有的PMI值
    pmi_values = []
    for tag1 in pmi_matrix.index:
        for tag2 in pmi_matrix.columns:
            if tag1 < tag2:  # 只取上三角矩阵，避免重复
                pmi_values.append((tag1, tag2, pmi_matrix.loc[tag1, tag2]))
    
    # 转换为DataFrame并排序
    collocations = pd.DataFrame(pmi_values, columns=['tag1', 'tag2', 'pmi'])
    return collocations.sort_values('pmi', ascending=False).head(n)

def analyze_hashtag_collocations(json_data):
    """
    主函数：分析hashtag共现关系
    """
    # 提取hashtags
    hashtag_lists = load_and_extract_hashtags(json_data)
    
    # 计算共现矩阵
    cooccurrence_matrix, unique_hashtags = calculate_cooccurrence_matrix(hashtag_lists)
    
    # 计算PMI
    pmi_matrix = calculate_pmi(cooccurrence_matrix)
    
    # 获取top collocations
    top_collocations = get_top_collocations(pmi_matrix)
    
    return {
        'cooccurrence_matrix': cooccurrence_matrix,
        'pmi_matrix': pmi_matrix,
        'top_collocations': top_collocations
    }

# 使用示例：
if __name__ == "__main__":
    json_data = '/Users/hongxuzhou/rt_data/27:Nov:2024/3_hashtags_cal_collo/endoflifecare.json'
    results = analyze_hashtag_collocations(json_data)
    print("Top Collocations:")
    print(results['top_collocations'])
    pass
    