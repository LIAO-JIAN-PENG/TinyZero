import re
import random
import ast

def extract_schedule(solution_str):
    """
    從解答文本中提取排程答案，取得 <answer> 與 </answer> 之間的內容。
    預期答案為列表（例如 [0, 1, 2]）或字串 "No feasible schedule"。
    """
    # 嘗試從不同格式中擷取
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None

def validate_schedule_format(schedule, n):
    """
    驗證排程格式：
      - 檢查是否為列表，且為 0 ~ n-1 的排列（每個作業僅能使用一次）。
    """
    try:
        if not isinstance(schedule, list):
            return False
        if sorted(schedule) != list(range(n)):
            return False
    except:
        return False
    return True

def simulate_schedule(schedule, jobs):
    """
    模擬排程，檢查按順序處理各作業時，其累積處理時間是否均未超過各作業截止時間。
    
    Args:
        schedule: 排程列表，內含作業索引
        jobs: 作業列表，每筆資料為 (processing_time, deadline)
    
    回傳 True 表示排程可行，否則 False。
    """
    current_time = 0
    for idx in schedule:
        p, d = jobs[idx]
        current_time += p
        if current_time > d:
            return False
    return True

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """
    EDF 排程問題的 reward 計算函式，評分階段分別如下：
    
    1. **擷取解答**：
       - 從 solution_str 擷取 <answer>...</answer> 內容，若失敗則回傳 0 分。
       
    2. **檢查排程格式**：
       - 若答案為 "No feasible schedule"：
         - 與 ground_truth['schedule'] 比對，若一致則滿分，否則僅得 format_score。
       - 若答案為列表，必須檢查是否為 0 ~ n-1 的排列，否則回傳 format_score。
       
    3. **模擬排程可行性**：
       - 根據 jobs 模擬排程，若累計處理時間超過某作業截止時間，則視為不可行，回傳 format_score。
       
    4. **最終評分**：
       - 當排程格式正確且可行，則回傳滿分 score。
    
    Args:
        solution_str: 模型生成的解答文本
        ground_truth: 字典，包含 "jobs" 與 "schedule" 兩個鍵
        format_score: 格式正確但結果錯誤給予的分數
        score: 正確答案給予的分數
    """
    jobs = ground_truth['jobs']
    expected = ground_truth['schedule']
    n = len(jobs)
    
    extracted = extract_schedule(solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print("Jobs:", jobs)
        print("Expected schedule:", expected)
        print("Extracted schedule:", extracted)
        print("Solution string:", solution_str)
        
    # 階段 1. 擷取解答失敗 → 0 分
    if extracted is None:
        if do_print:
            print("未能提取排程答案")
        return 0
    
    # 處理 "No feasible schedule" 的情況
    if extracted.strip() == "No feasible schedule":
        if expected == "No feasible schedule":
            if do_print:
                print("正確判斷無可行排程")
            return score
        else:
            if do_print:
                print("錯誤地判斷為無可行排程")
            return format_score
    
    # 階段 2. 嘗試解析答案為排程列表
    try:
        schedule = ast.literal_eval(extracted)
    except Exception as e:
        if do_print:
            print("解析排程錯誤:", e)
        return format_score
    
    # 驗證排程格式是否正確
    if not validate_schedule_format(schedule, n):
        if do_print:
            print("排程格式錯誤或缺少作業")
        return format_score
    
    # 階段 3. 模擬排程以驗證其可行性
    if not simulate_schedule(schedule, jobs):
        if do_print:
            print("排程模擬失敗，無法滿足所有截止時間")
        return format_score
    
    # 階段 4. 全部條件皆滿足 → 滿分
    if do_print:
        print("排程正確且可行:", schedule)
    return score
