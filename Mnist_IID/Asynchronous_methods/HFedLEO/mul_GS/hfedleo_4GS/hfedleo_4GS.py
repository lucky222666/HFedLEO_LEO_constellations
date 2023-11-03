from vir_estimate_comm import *
from vir_estimate_comp import *


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 在分数据集之前 先设计一个比较友好的随机种子
    setup_seed(20)
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train, dict_users_test = iid(dataset_train, dataset_test, args.num_users)
        else:
            dict_users_train, dict_users_test = noniid(dataset_train, dataset_test, args.num_users, 0.5)
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('../data/emnist/', split="bymerge", train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST('../data/emnist/', split="bymerge", train=False, download=True, transform=trans_emnist)
        # sample users
        if args.iid:
            dict_users_train, dict_users_test = iid(dataset_train, dataset_test, args.num_users)
        else:
            dict_users_train, dict_users_test = noniid(dataset_train, dataset_test, args.num_users, 0.5)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        # sample users
        if args.iid:
            dict_users_train, dict_users_test = iid(dataset_train, dataset_test, args.num_users)
        else:
            dict_users_train, dict_users_test = noniid(dataset_train, dataset_test, args.num_users, 0.5)
          
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = Res_Model(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'emnist':
        net_glob = CNNEmnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')


    # read file and get structure data, global parameters
    read_time_file_init("./input_info/sat_gs0_encounter_time.txt", 0) # 0-120
    read_time_file_add("./input_info/sat_gs1_encounter_time.txt", per_gs_visible_times * 1)
    read_time_file_add("./input_info/sat_gs2_encounter_time.txt", per_gs_visible_times * 2)
    read_time_file_add("./input_info/sat_gs3_encounter_time.txt", per_gs_visible_times * 3)

    read_range1_file_add("./input_info/sat_gs0_encounter_range.txt", 0)
    read_range1_file_add("./input_info/sat_gs1_encounter_range.txt", per_gs_visible_times * 1)
    read_range1_file_add("./input_info/sat_gs2_encounter_range.txt", per_gs_visible_times * 2)
    read_range1_file_add("./input_info/sat_gs3_encounter_range.txt", per_gs_visible_times * 3)

    read_range2_file_add()

    satellites = satellites[1:]
    sat_belongto_orbit()
    cal_taking_a_spin_around(orbits)

    # 统一所有轨道上卫星的索引下标，置为0
    vir_train_init()
    
    # 及时删除之前训练任务留下的文件
    if os.path.exists('./output_info/participants.txt'): 
        os.remove('./output_info/participants.txt')
    
    for iter in range(args.epochs+1):
        if os.path.exists('./save/model_w'+str(iter)+'.pth'): 
            os.remove('./save/model_w'+str(iter)+'.pth')

    fraction = 0.30
    alpha = 0.1
    next_epoch_start_time = stk_train_start_stamp
    pre_accuracy = 1

    # 记录那些没有及时更新的轨道
    history_selected_orbits = []

    # 提前训练好俩轮
    for iter in range(2):
        next_epoch_start_time = vir_comm_main(history_selected_orbits, fraction, next_epoch_start_time, iter)
        avg_acc, vir_global_acc = vir_comp_main(args, dataset_train, dataset_test, dict_users_train, dict_users_test, net_glob, fraction, (iter+1))       
        print("Epoch No." + str(iter+1) + " fraction is", fraction)
        print("Epoch No." + str(iter+1) + " training is over\n")
        # print("Average accuracy is:", avg_acc, ", virtual accuracy based on the global test set is:", vir_global_acc, "\n")

    fraction = fraction - alpha * (avg_acc - 0) / pre_accuracy
    pre_accuracy = avg_acc

    for iter in range(2, args.epochs):
        next_epoch_start_time = vir_comm_main(history_selected_orbits, fraction, next_epoch_start_time, iter)
        avg_acc, vir_global_acc = vir_comp_main(args, dataset_train, dataset_test, dict_users_train, dict_users_test, net_glob, fraction, (iter+1))        
        print("Epoch No." + str(iter+1) + " fraction is", fraction)
        print("Epoch No." + str(iter+1) + " training is over\n")
        # print("Average accuracy is:", avg_acc, ", virtual accuracy based on the global test set is:", vir_global_acc, "\n")
        fraction = fraction - alpha * (avg_acc - pre_accuracy) / pre_accuracy
        pre_accuracy = avg_acc

        if fraction < 0:
            print("fraction is less than 0 and a suitable orbit cannot be selected.")
            break
    