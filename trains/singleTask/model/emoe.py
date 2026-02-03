import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets.transformers_encoder.transformer import TransformerEncoder, MultimodalTransformer_w_JR
from ...singleTask.model.router import router
class EMOE(nn.Module):
    def __init__(self, args):
        super(EMOE, self).__init__()
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'biovid':
            # 各模态的序列长度
            if args.need_data_aligned:
                self.len_ecg, self.len_gsr, self.len_v = 75, 75, 75
            else:
                self.len_ecg, self.len_gsr, self.len_v = 8, 8, 75
        self.aligned = args.need_data_aligned
        # 原始特征维度
        self.orig_d_ecg, self.orig_d_gsr, self.orig_d_v = args.feature_dims
        # 模态目标特征维度相同
        self.d_ecg = self.d_gsr = self.d_v = dst_feature_dims
        # transformer多头注意力的头数
        self.num_heads = nheads
        # transformer编码器的层数
        self.layers = args.nlevels
        # 注意力机制的丢弃率
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_ecg = args.attn_dropout_ecg
        self.attn_dropout_gsr = args.attn_dropout_gsr
        # 各种丢弃率
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        # 注意力机制是否使用掩码
        self.attn_mask = args.attn_mask
        # 融合方法
        self.fusion_method = args.fusion_method
        output_dim = args.output_dim
        self.args = args
        # jmt的各参数
        self.jmt_nheads = args.jmt_nheads
        self.jmt_hidden_dim = args.jmt_hidden_dim
        self.jmt_num_layers = args.jmt_num_layers
        self.jmt_output_format = args.jmt_output_format
        self.jmt_dropout = args.jmt_dropout

        # 为各模态创建1D卷积投影层，将原始特征维度映射到目标维度
        self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=args.conv1d_kernel_size_ecg, padding=0, bias=False)
        self.proj_gsr = nn.Conv1d(self.orig_d_gsr, self.d_gsr, kernel_size=args.conv1d_kernel_size_gsr, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 创建1x1卷积编码器，为每一个时间步的特征进行变换
        self.encoder_c = nn.Conv1d(self.d_ecg, self.d_ecg, kernel_size=1, padding=0, bias=False)
        self.encoder_ecg = nn.Conv1d(self.d_ecg, self.d_ecg, kernel_size=1, padding=0, bias=False)
        self.encoder_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_gsr = nn.Conv1d(self.d_gsr, self.d_gsr, kernel_size=1, padding=0, bias=False)

        # 为单模态创建自注意力网络。
        self.self_attentions_ecg = self.get_network(self_type='ecg')
        self.self_attentions_v = self.get_network(self_type='v')
        self.self_attentions_gsr = self.get_network(self_type='gsr')

        #单模态预测头（两个线性投影层+一个最终输出层）
        self.proj1_ecg = nn.Linear(self.d_ecg, self.d_ecg)
        self.proj2_ecg = nn.Linear(self.d_ecg, self.d_ecg)
        self.out_layer_ecg = nn.Linear(self.d_ecg, output_dim)

        self.proj1_v = nn.Linear(self.d_ecg, self.d_ecg)
        self.proj2_v = nn.Linear(self.d_ecg, self.d_ecg)
        self.out_layer_v = nn.Linear(self.d_ecg, output_dim)

        self.proj1_gsr = nn.Linear(self.d_ecg, self.d_ecg)
        self.proj2_gsr = nn.Linear(self.d_ecg, self.d_ecg)
        self.out_layer_gsr = nn.Linear(self.d_ecg, output_dim)

        # JMT预测头
        self.multitransfomer = MultimodalTransformer_w_JR(
            ecg_dim=self.d_ecg,
            gsr_dim=self.d_gsr,
            v_dim=self.d_v,
            num_heads=self.jmt_nheads,
            hidden_dim=self.jmt_hidden_dim,
            num_layers=self.jmt_num_layers,
            output_format=self.jmt_output_format,
        )
        if self.jmt_output_format == "SELF_ATTEN":
            dim = dst_feature_dims
        elif self.jmt_output_format == "FC":
            dim = 1024
        # 2d JMT融合特征预测头
        self.out_layer_c = nn.Sequential(nn.Linear(dim, 128),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(self.jmt_dropout),
                                        nn.Linear(128, output_dim)
                                        )
        # 3d JMT融合特征预测头
        # self.proj1_c = nn.Linear(self.d_ecg, self.d_ecg)
        # self.proj2_c = nn.Linear(self.d_ecg, self.d_ecg)
        # self.out_layer_c = nn.Linear(self.d_ecg, output_dim)

        # 路由网络，计算权重W
        # 原始-router
        # self.Router = router(self.orig_d_ecg * self.len_v + self.orig_d_gsr * self.len_v + self.orig_d_v * self.len_v, 3, self.args.temperature)
        # SE-router
        self.Router = router(3*self.len_v, 3, self.args.temperature)
        # SE+原始-router
        # self.Router = router(self.len_v, 3, self.args.temperature, self.orig_d_ecg * self.len_v + self.orig_d_gsr * self.len_v + self.orig_d_v * self.len_v)
        # 将ecg、gsr序列长度对齐到视觉序列长度
        self.transfer_ecg_ali = nn.Linear(self.len_ecg, self.len_v)
        self.transfer_gsr_ali = nn.Linear(self.len_gsr, self.len_v)


    def get_network(self, self_type='ecg', layers=-1):
        if self_type == 'ecg':
            embed_dim, attn_dropout = self.d_ecg, self.attn_dropout_ecg
        elif self_type == 'gsr':
            embed_dim, attn_dropout = self.d_gsr, self.attn_dropout_gsr
        elif self_type == 'v':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def get_net(self, name):
        return getattr(self, name)

    def forward(self, ecg, gsr, video):

        # 将序列长度和特征维度交换(batch, feature, seq)
        x_ecg = ecg.transpose(1, 2)
        x_gsr = gsr.transpose(1, 2)
        x_v = video.transpose(1, 2)

        # 原始
        # if not self.aligned:
        #     # 未对齐，使用线性层对齐序列长度(batch, seq, feature)
        #     ecg_ = self.transfer_ecg_ali(ecg.permute(0, 2, 1)).permute(0, 2, 1)
        #     gsr_ = self.transfer_gsr_ali(gsr.permute(0, 2, 1)).permute(0, 2, 1)
        #     m_i = torch.cat((ecg_, gsr_, video), dim=2)
        # else:
        #     m_i = torch.cat((ecg, gsr, video), dim=2)
        # # 路由网络获取权重
        # m_w = self.Router(m_i)

        # SE
        if not self.aligned:
            # 未对齐，使用线性层对齐序列长度 (batch, seq, feature)
            ecg_ = self.transfer_ecg_ali(ecg.permute(0, 2, 1)).permute(0, 2, 1)
            gsr_ = self.transfer_gsr_ali(gsr.permute(0, 2, 1)).permute(0, 2, 1)
            m_w = self.Router(ecg_, gsr_, video)
        else:
            m_w = self.Router(ecg, gsr, video)

        # 如果原始维度与目标维度不同，进行投影（此时seq维度也发生了变化）
        proj_x_ecg = x_ecg if self.orig_d_ecg == self.d_ecg else self.proj_ecg(x_ecg)
        proj_x_gsr = x_gsr if self.orig_d_gsr == self.d_gsr else self.proj_gsr(x_gsr)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # 使用encoder（1*1卷积）对投影后特征编码，得到低级特征
        c_ecg = self.encoder_c(proj_x_ecg)
        c_v = self.encoder_c(proj_x_v)
        c_gsr = self.encoder_c(proj_x_gsr)
        # 交换维度(seq, batch, feat)
        c_ecg = c_ecg.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_gsr = c_gsr.permute(2, 0, 1)

        # 对每个模态应用transformer，得到高级特征
        c_ecg_att_seq = self.self_attentions_ecg(c_ecg) # (seq, batch, feat)
        if type(c_ecg_att_seq) == tuple:
            c_ecg_att_seq = c_ecg_att_seq[0]
        c_ecg_att = c_ecg_att_seq[-1] # (batch, feat)

        c_v_att_seq = self.self_attentions_v(c_v)
        if type(c_v_att_seq) == tuple:
            c_v_att_seq = c_v_att_seq[0]
        c_v_att = c_v_att_seq[-1]

        c_gsr_att_seq = self.self_attentions_gsr(c_gsr)
        if type(c_gsr_att_seq) == tuple:
            c_gsr_att_seq = c_gsr_att_seq[0]
        c_gsr_att = c_gsr_att_seq[-1]

        # ecg模态预测结果
        ecg_proj = self.proj2_ecg(
            F.dropout(
                F.relu(
                    self.proj1_ecg(c_ecg_att), inplace=True), p=self.output_dropout, training=self.training))
        ecg_proj += c_ecg_att
        logits_ecg = self.out_layer_ecg(ecg_proj)
        # 视觉模态预测结果
        v_proj = self.proj2_v(
            F.dropout(
                F.relu(
                    self.proj1_v(c_v_att), inplace=True), p=self.output_dropout, training=self.training))
        v_proj += c_v_att
        logits_v = self.out_layer_v(v_proj)
        # gsr模态预测结果
        gsr_proj = self.proj2_gsr(
            F.dropout(
                F.relu(
                    self.proj1_gsr(c_gsr_att), inplace=True), p=self.output_dropout, training=self.training))
        gsr_proj += c_gsr_att
        logits_gsr = self.out_layer_gsr(gsr_proj)

        # 加权融合模态预测结果
        ## 3d张量
        # ecg_weights = m_w[:, 0].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        # gsr_weights = m_w[:, 1].unsqueeze(1).unsqueeze(2)
        # v_weights = m_w[:, 2].unsqueeze(1).unsqueeze(2)
        # w_ecg = c_ecg_att_seq.permute(1, 0, 2) * ecg_weights
        # w_gsr = c_gsr_att_seq.permute(1, 0, 2) * gsr_weights
        # w_v = c_v_att_seq.permute(1, 0, 2) * v_weights
        # ## 3d预测头
        # c_att_seq = self.multitransfomer(w_ecg, w_gsr, w_v).permute(1, 0, 2) # (seq, batch, feat)
        # c_att = c_att_seq[-1] # (batch, feat)
        # c_proj = self.proj2_c(
        #     F.dropout(
        #         F.relu(
        #             self.proj1_c(c_att), inplace=True), p=self.output_dropout, training=self.training))
        # c_proj += c_att
        # logits_c = self.out_layer_c(c_proj)

        ## 2d张量
        ecg_weights = m_w[:, 0].view(-1, 1)
        gsr_weights = m_w[:, 1].view(-1, 1)
        v_weights = m_w[:, 2].view(-1, 1)
        w_ecg = c_ecg_att * ecg_weights
        w_gsr = c_gsr_att * gsr_weights
        w_v = c_v_att * v_weights
        ## 2d预测头
        c_proj = self.multitransfomer(w_ecg, w_gsr, w_v)# (batch, feat)/(batch, 1024)\
        c_proj += c_ecg_att + c_gsr_att + c_v_att
        logits_c = self.out_layer_c(c_proj)


        res = {
            'logits_c': logits_c,
            'logits_ecg': logits_ecg,
            'logits_v': logits_v,
            'logits_gsr': logits_gsr,
            'channel_weight': m_w,
            'c_proj': c_proj,
            'ecg_proj': ecg_proj,
            'v_proj': v_proj,
            'gsr_proj': gsr_proj,
            # 'c_fea': c_fusion,
        }
        return res