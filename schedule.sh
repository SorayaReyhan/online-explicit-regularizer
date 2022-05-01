# MNIST
CUDA_VISIBLE_DEVICES="0" python train.py --dataset=mnist --num_tasks=10 --num_classes=10 --model=mnist
CUDA_VISIBLE_DEVICES="0" python train.py --dataset=mnist --num_tasks=10 --num_classes=10 --model=mnist --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --dataset=mnist --num_tasks=10 --num_classes=10 --model=mnist --trainer=online_explicit_ewc

# MNIST DROPOUT
CUDA_VISIBLE_DEVICES="0" python train.py --dataset=mnist --num_tasks=10 --num_classes=10 --model=mnist --dropout=0.1 --name=dropout
CUDA_VISIBLE_DEVICES="0" python train.py --dataset=mnist --num_tasks=10 --num_classes=10 --model=mnist --dropout=0.1 --name=dropout --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --dataset=mnist --num_tasks=10 --num_classes=10 --model=mnist --dropout=0.1 --name=dropout --trainer=online_explicit_ewc

# Malware
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla --trainer=online_explicit_ewc

# Malware DROPOUT
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla --dropout=0.1 --name=dropout
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla --dropout=0.1 --name=dropout --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla --dropout=0.1 --name=dropout --trainer=online_explicit_ewc

# CIFAR100
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=cifar100 --num_tasks=5 --num_classes=10 --model=vanilla_cifar
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=cifar100 --num_tasks=5 --num_classes=10 --model=vanilla_cifar --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=cifar100 --num_tasks=5 --num_classes=10 --model=vanilla_cifar --trainer=online_explicit_ewc

# CIFAR100 DROPOUT
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=cifar100 --num_tasks=5 --num_classes=10 --model=vanilla_cifar --dropout=0.1 --name=dropout
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=cifar100 --num_tasks=5 --num_classes=10 --model=vanilla_cifar --dropout=0.1 --name=dropout --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=cifar100 --num_tasks=5 --num_classes=10 --model=vanilla_cifar --dropout=0.1 --name=dropout --trainer=online_explicit_ewc

# Malware Vanill_cnn
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla_cnn
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla_cnn --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla_cnn --trainer=online_explicit_ewc

# Malware Vanill_cnn
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla_cnn --dropout=0.1 --name=vanilla_cnn_drop
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla_cnn --dropout=0.1 --name=vanilla_cnn_drop --trainer=ewc
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=100 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla_cnn --dropout=0.1 --name=vanilla_cnn_drop --trainer=online_explicit_ewc


# Malware EWC hyperparameter search for importance
CUDA_VISIBLE_DEVICES="0" python train.py --epochs=50 --dataset=malware --num_tasks=4 --num_classes=5 --model=vanilla --name=ewc_hp --trainer=ewc --importance=10000
