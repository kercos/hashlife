import torch

'''
    Test some alternative methods (sum instead of conv2d)
'''
def test_sum_pool():
    input = torch.ones((5,5), dtype=torch.float16)
    output = torch.nn.functional.avg_pool1d(input, 3, stride=1) * 8
    print(output)

if __name__ == "__main__":
    test_sum_pool()