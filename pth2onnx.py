import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime
from trains.singleTask.model.emoe import EMOE


# -----------------------------------#
#   EMOE模型转换为ONNX格式
# -----------------------------------#
def emoe_model_convert_onnx(model, args, output_path):
    # 根据对齐情况创建示例输入
    if args.need_data_aligned:
        # 对齐数据 - 所有模态序列长度相同
        batch_size = 1
        seq_len = 75
        dummy_ecg = torch.randn(batch_size, seq_len, args.feature_dims[0], dtype=torch.float32)
        dummy_gsr = torch.randn(batch_size, seq_len, args.feature_dims[1], dtype=torch.float32)
        dummy_video = torch.randn(batch_size, seq_len, args.feature_dims[2], dtype=torch.float32)
    else:
        # 未对齐数据 - 序列长度不同
        batch_size = 1
        dummy_ecg = torch.randn(batch_size, 8, args.feature_dims[0], dtype=torch.float32)
        dummy_gsr = torch.randn(batch_size, 8, args.feature_dims[1], dtype=torch.float32)
        dummy_video = torch.randn(batch_size, 75, args.feature_dims[2], dtype=torch.float32)

    input_names = ["ecg", "gsr", "video"]  # 输入节点名称
    output_names = ["logits_c", "logits_ecg", "logits_v", "logits_gsr", "channel_weight"]  # 输出节点名称

    torch.onnx.export(
        model,
        (dummy_ecg, dummy_gsr, dummy_video),
        output_path,
        verbose=True,  # 显示详细信息
        keep_initializers_as_inputs=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'ecg': {0: 'batch_size'},
            'gsr': {0: 'batch_size'},
            'video': {0: 'batch_size'},
            'logits_c': {0: 'batch_size'},
            'logits_ecg': {0: 'batch_size'},
            'logits_v': {0: 'batch_size'},
            'logits_gsr': {0: 'batch_size'},
            'channel_weight': {0: 'batch_size'}
        }
    )


# -----------------------------------#
#   加载训练好的EMOE模型
# -----------------------------------#
def load_emoe_model(model_path, args):
    # 创建模型结构
    model = EMOE(args)

    # 加载训练好的参数
    checkpoint = torch.load(model_path, map_location='cpu')

    # 根据checkpoint类型加载参数
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("加载模型状态字典完成")
        else:
            model.load_state_dict(checkpoint)
            print("直接加载状态字典完成")
    else:
        model = checkpoint
        print("加载整个模型对象完成")

    model.eval()
    return model


# -----------------------------------#
#   验证ONNX模型与PyTorch模型的一致性
# -----------------------------------#
def validate_onnx_model(original_model, onnx_path, args):
    # 创建测试输入
    if args.need_data_aligned:
        batch_size = 2
        seq_len = 75
        test_ecg = torch.randn(batch_size, seq_len, args.feature_dims[0], dtype=torch.float32)
        test_gsr = torch.randn(batch_size, seq_len, args.feature_dims[1], dtype=torch.float32)
        test_video = torch.randn(batch_size, seq_len, args.feature_dims[2], dtype=torch.float32)
    else:
        batch_size = 2
        test_ecg = torch.randn(batch_size, 8, args.feature_dims[0], dtype=torch.float32)
        test_gsr = torch.randn(batch_size, 8, args.feature_dims[1], dtype=torch.float32)
        test_video = torch.randn(batch_size, 75, args.feature_dims[2], dtype=torch.float32)

    # PyTorch模型推理
    with torch.no_grad():
        torch_outputs = original_model(test_ecg, test_gsr, test_video)

    # 提取主要输出
    torch_logits_c = torch_outputs['logits_c']
    torch_logits_ecg = torch_outputs['logits_ecg']
    torch_logits_v = torch_outputs['logits_v']
    torch_logits_gsr = torch_outputs['logits_gsr']
    torch_channel_weight = torch_outputs['channel_weight']

    # ONNX模型推理
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # 准备ONNX输入
    ort_inputs = {
        ort_session.get_inputs()[0].name: test_ecg.numpy(),
        ort_session.get_inputs()[1].name: test_gsr.numpy(),
        ort_session.get_inputs()[2].name: test_video.numpy()
    }

    # ONNX推理
    ort_outputs = ort_session.run(None, ort_inputs)

    # 提取ONNX输出
    onnx_logits_c = ort_outputs[0]
    onnx_logits_ecg = ort_outputs[1]
    onnx_logits_v = ort_outputs[2]
    onnx_logits_gsr = ort_outputs[3]
    onnx_channel_weight = ort_outputs[4]

    print("PyTorch模型输出形状:")
    print(f"logits_c: {torch_logits_c.shape}")
    print(f"logits_ecg: {torch_logits_ecg.shape}")
    print(f"logits_v: {torch_logits_v.shape}")
    print(f"logits_gsr: {torch_logits_gsr.shape}")
    print(f"channel_weight: {torch_channel_weight.shape}")

    print("\nONNX模型输出形状:")
    print(f"logits_c: {onnx_logits_c.shape}")
    print(f"logits_ecg: {onnx_logits_ecg.shape}")
    print(f"logits_v: {onnx_logits_v.shape}")
    print(f"logits_gsr: {onnx_logits_gsr.shape}")
    print(f"channel_weight: {onnx_channel_weight.shape}")

    # 验证输出一致性
    try:
        np.testing.assert_allclose(torch_logits_c.numpy(), onnx_logits_c, rtol=1e-03, atol=1e-05)
        print("✓ logits_c 输出一致")
    except AssertionError as e:
        print(f"✗ logits_c 输出不一致: {e}")

    try:
        np.testing.assert_allclose(torch_logits_ecg.numpy(), onnx_logits_ecg, rtol=1e-03, atol=1e-05)
        print("✓ logits_ecg 输出一致")
    except AssertionError as e:
        print(f"✗ logits_ecg 输出不一致: {e}")

    try:
        np.testing.assert_allclose(torch_logits_v.numpy(), onnx_logits_v, rtol=1e-03, atol=1e-05)
        print("✓ logits_v 输出一致")
    except AssertionError as e:
        print(f"✗ logits_v 输出不一致: {e}")

    try:
        np.testing.assert_allclose(torch_logits_gsr.numpy(), onnx_logits_gsr, rtol=1e-03, atol=1e-05)
        print("✓ logits_gsr 输出一致")
    except AssertionError as e:
        print(f"✗ logits_gsr 输出不一致: {e}")

    try:
        np.testing.assert_allclose(torch_channel_weight.numpy(), onnx_channel_weight, rtol=1e-03, atol=1e-05)
        print("✓ channel_weight 输出一致")
    except AssertionError as e:
        print(f"✗ channel_weight 输出不一致: {e}")


if __name__ == '__main__':
    # 定义模型参数
    class Args:
        def __init__(self):
            self.dataset_name = 'biovid'
            self.need_data_aligned = False  # 根据你的模型训练设置调整
            self.feature_dims = (22, 39, 35)
            self.dst_feature_dim_nheads = (256, 8)
            self.nlevels = 4
            self.attn_dropout_v = 0.1
            self.attn_dropout_ecg = 0.2
            self.attn_dropout_gsr = 0.1
            self.relu_dropout = 0.0
            self.embed_dropout = 0.2
            self.res_dropout = 0.0
            self.output_dropout = 0.5
            self.attn_mask = False  # 为避免triu问题，暂时禁用
            self.fusion_method = 'sum'
            self.output_dim = 5
            self.conv1d_kernel_size_ecg = 5
            self.conv1d_kernel_size_gsr = 5
            self.conv1d_kernel_size_v = 5
            self.jmt_nheads = 8
            self.jmt_hidden_dim = 256
            self.jmt_num_layers = 2
            self.jmt_output_format = "SELF_ATTEN"
            self.jmt_dropout = 0.1
            self.temperature = 0.1


    args = Args()

    # 模型路径和输出路径
    model_path = "./pt/emoe.pth"
    onnx_output_path = './pt/emoe.onnx'

    # 1. 加载训练好的模型
    print("正在加载EMOE模型...")
    emoe_model = load_emoe_model(model_path, args)
    print("模型加载完成")

    # 2. 导出为ONNX模型
    print("正在导出ONNX模型...")
    emoe_model_convert_onnx(emoe_model, args, onnx_output_path)
    print("EMOE模型转换为ONNX完成")

    # 3. 第一轮验证：检查ONNX模型格式
    print("\n正在进行ONNX模型格式验证...")
    try:
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型格式验证通过")
    except Exception as e:
        print(f"✗ ONNX模型格式验证失败: {e}")

    # 4. 第二轮验证：验证推理一致性
    print("\n正在进行推理一致性验证...")
    validate_onnx_model(emoe_model, onnx_output_path, args)

    print("\n所有步骤完成！")