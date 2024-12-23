# LAP-for-LLM

拟修改大模型训练时使用的优化器，应用最小作用量原理。

拟基于unsloth推理训练框架的源码修改，可能继续加入trl等库的源码用于修改优化器层面的逻辑。

## 配环境

本机配置：

> GPU: NVIDIA GeForce RTX 3060 Laptop GPU. Max memory: 6.0 GB. Platform: Windows.
> Torch: 2.4.0+cu121. CUDA: 8.6. CUDA Toolkit: 12.1. Triton: 3.1.0
> Bfloat16 = TRUE. FA [Xformers = 0.0.27.post2. FA2 = False]

1. 拉取仓库

     ```
     git clone https://github.com/Katock-Cricket/LAP-for-LLM.git
     ```

2. 创建conda环境

     ```
     conda create --name LAP python=3.10
     ```

3. 安装ninja

     ```
     pip install ninja
     ```

4. 安装triton

     ```
     pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp310-cp310-win_amd64.whl
     ```

5. 安装bnb

     ```
     pip install bitsandbytes
     ```

6. 安装trl, datasets

     ```
     pip install trl
     pip install datasets
     ```

7. 安装xformers、torch、torchvision、torchaudio

     ```
     pip3 install -U xformers torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

## 验证环境

1. 验证ninja，输出False/0

     ```
     echo %?
     ```

2. 执行test_env脚本，验证CUDA可用

     ```
     python test_env.py
     ```

     ```
     E:\Anaconda\envs\LAP-for-LLM\python.exe D:\LearningMaterials\大模型\LAP-for-LLM\test_env.py 
     True
     12.1
     <module 'torch.version' from 'E:\\Anaconda\\envs\\LAP-for-LLM\\lib\\site-packages\\torch\\version.py'>
     D:\NVIDIA GPU Computing Toolkit\CUDA\v12.3
     ```

3. 验证xformers可用

     ```
     python -m xformers.info
     ```

     ```
     xFormers 0.0.27.post2
     ...
     ```

4. 验证bnb可用

     ```
     python -m bitsandbytes
     ```

     ```
     Checking that the library is importable and CUDA is callable...
     SUCCESS!
     Installation was successful!
     ```

## 准备基座模型和数据集

```
python prepare.py
```

数据集放在了./dataset

转换格式后的数据集放在了./dataset_conv

基座模型放在了./weights

## 开始训练

```
python train.py
```

```

```

流程参考：https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing
