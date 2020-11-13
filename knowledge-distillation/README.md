### Running the knowledge distillation experiments

Most of the procedure is identical to that in [https://github.com/HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller).


1. Fetch the pretrained teacher models by:
    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run the following command to train the distillation model
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ${method} --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --cpc_alpha 1.0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method (such as `crd`, `cpc`, `ml_cpc`)
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    - `--cpc_alpha`: specify the `alpha` value used in CPC and ML-CPC (these two options only)
    
3. View the results in `save/student_tensorboards` directory.