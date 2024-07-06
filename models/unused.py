class EquivariantGraphLSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(EquivariantGraphLSTM, self).__init__()
        # 定义每个门的等变卷积层
        self.conv_x_i = Convolution(...) # 输入门的卷积层
        self.conv_h_i = Convolution(...) # 输入门的历史卷积层
        self.conv_x_f = Convolution(...) # 遗忘门的卷积层
        self.conv_h_f = Convolution(...) # 遗忘门的历史卷积层
        self.conv_x_c = Convolution(...) # 细胞状态的卷积层
        self.conv_h_c = Convolution(...) # 细胞状态的历史卷积层
        self.conv_x_o = Convolution(...) # 输出门的卷积层
        self.conv_h_o = Convolution(...) # 输出门的历史卷积层

        # 权重和偏置初始化
        self.w_c_i = torch.nn.Parameter(torch.Tensor(out_channels))
        self.w_c_f = torch.nn.Parameter(torch.Tensor(out_channels))
        self.w_c_o = torch.nn.Parameter(torch.Tensor(out_channels))
        self.w_c = torch.nn.Parameter(torch.Tensor(out_channels))

        self.b_i = torch.nn.Parameter(torch.Tensor(out_channels))
        self.b_f = torch.nn.Parameter(torch.Tensor(out_channels))
        self.b_c = torch.nn.Parameter(torch.Tensor(out_channels))
        self.b_o = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x, h, c):
        # 应用卷积并计算 LSTM 门的激活
        i = torch.sigmoid(self.conv_x_i(x) + self.conv_h_i(h) + self.w_c_i * c + self.b_i)
        f = torch.sigmoid(self.conv_x_f(x) + self.conv_h_f(h) + self.w_c_f * c + self.b_f)
        o = torch.sigmoid(self.conv_x_o(x) + self.conv_h_o(h) + self.w_c_o * c + self.b_o)
        c_new = f * c + i * torch.tanh(self.conv_x_c(x) + self.conv_h_c(h) + self.b_c)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new
