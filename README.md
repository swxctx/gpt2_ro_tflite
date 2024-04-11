- 模型转换

    ```shell
    
    mkdir model
    
    cd model

    # lfs 安装
    git lfs install

    # 下载模型
    git clone https://huggingface.co/openai-community/gpt2

    # 将PyTorch模型转换为ONNX
    python3 convert_p_to_onnx.py

    # 将ONNX模型转换为TensorFlow模型
    python3 convert_onnx_to_tf.py

    # 将TensorFlow模型转换为TensorFlow Lite模型
    python3 convert_tf_to_tflite.py
    ```
    
 - 分词

 	```
 	python3 fenci.py 获得input id
 	python3 result.py 可以通过生成结果id转换得到实际文本
 	```