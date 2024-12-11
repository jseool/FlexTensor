import argparse
import json
import re

def extract_numbers_from_schedule(schedule):
    """
    schedule 문자열에서 모든 숫자를 추출하여 리스트로 반환.
    """
    # 정규식을 사용해 숫자를 추출 (정수와 소수 포함)
    numbers = re.findall(r'-?\d+\.?\d*', schedule)
    # 문자열로 추출된 숫자를 float으로 변환
    return [float(num) if '.' in num else int(num) for num in numbers]

def convert_gemm_log_to_trainable_format(input_file, output_file):
    """
    gemm.log 데이터를 train_performance_model에서 사용할 수 있는 형식으로 변환하여 JSONL로 저장
    """
    dataset = []
    with open(input_file, "r") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if line:  # 빈 줄 건너뜀
                try:
                    # ':'로 구분하여 이름과 나머지 데이터 분리
                    name, rest = line.split(":", 1)
                    # ']'와 실행 시간을 분리
                    schedule, time = rest.rsplit("]:", 1)
                    time = float(time.strip())  # 실행 시간을 float으로 변환
                    
                    # schedule 문자열에서 숫자만 추출
                    input_features = extract_numbers_from_schedule(schedule)
                    
                    # 숫자 리스트와 실행 시간을 튜플로 데이터셋에 추가
                    dataset.append((input_features, [time]))
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")

    # 데이터를 JSONL 포맷으로 저장
    with open(output_file, "w") as fout:
        for data in dataset:
            fout.write(json.dumps(list(data)) + "\n")
    print(f"변환 완료! 저장 위치: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gemm.log 데이터를 train_performance_model 형식으로 변환")
    parser.add_argument("input_file", type=str, help="변환할 gemm.log 파일 경로")
    parser.add_argument("output_file", type=str, help="저장할 JSONL 파일 경로")

    args = parser.parse_args()
    convert_gemm_log_to_trainable_format(args.input_file, args.output_file)