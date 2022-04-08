import torch
import torch.nn.functional as F
import numpy as np


def compute_h3_answer(out, top_k, label_decode):
    probs = F.softmax(out, dim=1).squeeze()
    probs, indices = probs.topk(top_k, largest=True)  # 选取概率最大的前top_k个
    indices = indices.cpu().numpy()
    probs = probs.cpu().numpy()
    char_index = np.random.choice(indices, size=1, p=probs / probs.sum())  # 随机选取一个索引
    if not isinstance(char_index, np.ndarray):  # 这里没太搞懂
        char_index = [char_index]
    h3_value = label_decode(char_index)
    # predict = h3_t_geo(h3_value.tolist())
    return h3_value


def train_epoch(net, train_iter, loss_function, optimizer, device, top_k, label_decode, label_encode, encoder):
    loss_ = 0
    how_many_instance = 0
    how_many_instance_right = 0
    state = None
    for i, k in train_iter[:-1]:  # 这里的i和k都还是h3编码
        k_origin = k
        if state is None:
            state = net.begin_state(batch_size=1, device=device)
        else:
            for s in state:
                s.detach_()
        optimizer.zero_grad()
        a = encoder.transform(i.reshape(-1, 1))
        a = torch.tensor(a).to(torch.float32).to(device)
        k = label_encode(k)
        k = torch.tensor(k).to(torch.long).to(device)
        out, state = net(a, state)
        loss = loss_function(out, k)
        loss.backward()
        optimizer.step()
        loss_ = loss_ + loss
        how_many_instance = how_many_instance + 1
        predict = compute_h3_answer(out.detach(), top_k, label_decode)  # 这个char——index和k完全不是一个东西
        # print(f"predict : {predict} k :  {k}")#这个一直不对
        if predict == k_origin:  # 都是h3的形式
            how_many_instance_right = how_many_instance_right + 1

    return how_many_instance, loss_, how_many_instance_right


def predict_epoch(net, test_iter, device, top_k, label_decode, encoder):
    predict_answer_list = []
    state = None
    how_many_instance = 0
    how_many_instance_right = 0
    with torch.no_grad():
        for i, k in test_iter[:-1]:
            k_origin = k
            if state is None:
                state = net.begin_state(batch_size=1, device=device)
            else:
                for s in state:
                    s.detach_()
            a = encoder.transform(i.reshape(-1, 1))
            a = torch.tensor(a).to(torch.float32).to(device)
            # k = label_encode(k)
            # k = torch.tensor(k).to(torch.long).to(device)  # k的size确实是1，然后out的size是884的向量

            out, state = net(a, state)

            predict_answer = compute_h3_answer(out, top_k, label_decode)
            # print(f"char_index : {char_index} k :  {k}")这个一直不对
            if predict_answer == k_origin:
                how_many_instance_right = how_many_instance_right + 1
            how_many_instance = how_many_instance + 1
            predict_answer_list.append(predict_answer)
            # 这个取索引的方法是取出前5个然后只根据这5个概率去看
    return how_many_instance, how_many_instance_right, predict_answer_list
