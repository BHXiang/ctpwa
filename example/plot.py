import uproot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import argparse
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages

def get_th1f_histograms(directory):
    """从TDirectory中提取所有TH1F直方图"""
    histograms = {}
    for key in directory.keys():
        if 'TH1F' in str(type(directory[key])):
            # 去除;1后缀
            clean_key = key.split(';')[0]
            histograms[clean_key] = directory[key]
    return histograms

def get_th2f_histograms(directory):
    """从TDirectory中提取所有TH2F直方图"""
    histograms = {}
    for key in directory.keys():
        if 'TH2F' in str(type(directory[key])):
            # 去除;1后缀
            clean_key = key.split(';')[0]
            histograms[clean_key] = directory[key]
    return histograms

def get_labels(directory):
    """从TDirectory中提取xlabel和ylabel"""
    xlabel, ylabel = "", ""
    for key in directory.keys():
        if 'xlabel' in key:
            xlabel_obj = directory[key]
            if hasattr(xlabel_obj, 'decode'):
                xlabel = xlabel_obj.decode('utf-8')
            else:
                xlabel = str(xlabel_obj)
        elif 'ylabel' in key:
            ylabel_obj = directory[key]
            if hasattr(ylabel_obj, 'decode'):
                ylabel = ylabel_obj.decode('utf-8')
            else:
                ylabel = str(ylabel_obj)
    return xlabel, ylabel

def read_legends(file):
    """从legends TTree中读取图例名称"""
    if 'legends' not in file:
        print("警告: 未找到legends TTree")
        return []
    
    try:
        # 读取legends树
        legends_tree = file['legends']
        legend_branch = legends_tree['legend']
        legends_array = legend_branch.array()
        
        if len(legends_array) == 0:
            print("警告: legends树中没有数据")
            return []
        
        # 第一个条目包含所有图例
        legend_list = legends_array[0]
        print(f"legends中读取到 {len(legend_list)} 个图例")
        
        # 将awkward数组转换为Python列表，确保正确解码字符串
        python_legend_list = []
        for item in legend_list:
            # 处理可能的bytes对象，类似get_labels函数中的处理
            if hasattr(item, 'decode'):
                try:
                    decoded_item = item.decode('utf-8')
                    python_legend_list.append(decoded_item)
                except Exception as decode_error:
                    print(f"解码图例项时出错: {decode_error}, 使用字符串表示")
                    python_legend_list.append(str(item))
            else:
                python_legend_list.append(str(item))

        return python_legend_list
    except Exception as e:
        print(f"读取legends时出错: {e}")
        return []

def determine_subplot_layout(n_plots):
    """根据子图数量确定最优布局"""
    if n_plots <= 0:
        return 0, 0
    elif n_plots == 1:
        return 1, 1
    elif n_plots == 2:
        return 1, 2  # 1行2列
    elif n_plots == 3:
        return 1, 3  # 1行3列
    elif n_plots == 4:
        return 2, 2  # 2行2列
    elif n_plots <= 6:
        return 2, 3  # 2行3列
    elif n_plots <= 9:
        return 3, 3  # 3行3列
    else:
        n_cols = min(4, ceil(n_plots ** 0.5))
        n_rows = ceil(n_plots / n_cols)
        return n_rows, n_cols

def create_legend_column(fig, all_component_plots, all_component_labels, legend_list=None, ax_legend=None, has_background=False):
    """在右侧创建图例列"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    
    # 收集所有唯一的组件标签
    unique_component_labels = []
    unique_component_plots = []
    
    # 定义线型列表，与绘图函数中的保持一致
    line_styles_list = ['-', '--', '-.', ':']  # 实线、虚线、点划线、点线
    
    # 如果提供了legend_list，直接使用
    if legend_list is not None and len(legend_list) > 0:
        # 为每个图例项创建代理plot对象
        proxy_plots = []
        num_legends = len(legend_list)
        if num_legends <= 20:
            # 使用tab20颜色映射，均匀分布
            colors = plt.cm.tab20(np.linspace(0, 1, num_legends))
            linestyles = ['-'] * num_legends  # 所有使用实线
        else:
            # 超过20种，循环使用颜色和线型
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = []
            linestyles = []
            for i in range(num_legends):
                # 循环使用颜色
                color_idx = i % 20
                # 每20种颜色换一种线型
                linestyle_idx = (i // 20) % len(line_styles_list)
                colors.append(base_colors[color_idx])
                linestyles.append(line_styles_list[linestyle_idx])

        for i, label in enumerate(legend_list):
            color = colors[i]
            linestyle = linestyles[i]

            # 创建代理对象
            proxy = Line2D([0], [0], color=color,
                          linewidth=1, alpha=0.7,
                          linestyle=linestyle, label=label)
            proxy_plots.append(proxy)
            unique_component_labels.append(label)
            unique_component_plots.append(proxy)
    else:
        # 使用实际绘制的组件
        for plots, labels in zip(all_component_plots, all_component_labels):
            for plot, label in zip(plots, labels):
                if label not in unique_component_labels:
                    unique_component_labels.append(label)
                    unique_component_plots.append(plot)
    
    # 创建自定义的error bar图例句柄类
    class ErrorBarHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            # 获取颜色
            color = 'black'
            if hasattr(orig_handle, 'get_color'):
                color = orig_handle.get_color()
            
            # 创建一个组合的图例项
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            
            # 创建三个部分：中心点、垂直线、顶部和底部的水平线
            center_x = x0 + width / 2
            center_y = y0 + height / 2
            
            # 创建垂直线（误差棒主体）
            line_vert = Line2D([center_x, center_x], 
                             [center_y - height/4, center_y + height/4],
                             color=color, linewidth=0.5,
                             transform=handlebox.get_transform())
            
            # 创建中心点（数据点）
            center_point = Line2D([center_x], [center_y],
                                 color=color, marker='o',
                                 markersize=1,  # 比实际绘图小一点
                                 linestyle='',
                                 transform=handlebox.get_transform())
            
            # 将所有部分添加到handlebox
            handlebox.add_artist(line_vert)
            handlebox.add_artist(center_point)
            
            return [line_vert, center_point]
    
    # 添加主要图例项（兼容有无h_bkg）
    if has_background:
        main_labels = ['Data', 'Fit+Bkg', 'Background']
        main_colors = ['black', 'red', 'gray']
        main_styles = ['errorbar', '-', 'fill']
    else:
        main_labels = ['Data', 'Fit']
        main_colors = ['black', 'red']
        main_styles = ['errorbar', '-']

    legend_handles = []
    legend_labels = []

    # 添加主要项
    for i, (label, color, style) in enumerate(zip(main_labels, main_colors, main_styles)):
        if label == 'Background':
            # 对于背景，使用填充的矩形作为图例句柄
            proxy = Patch(facecolor=color, alpha=0.3, edgecolor='none', label=label)
            legend_handles.append(proxy)
            legend_labels.append(label)
        elif label == 'Data':
            # 创建一个空的Line2D对象作为error bar的句柄
            # 实际的绘制将由ErrorBarHandler处理
            error_proxy = Line2D([], [], color=color, linewidth=0.8, label=label)
            legend_handles.append(error_proxy)
            legend_labels.append(label)
        else:  # Fit or Fit+Bkg
            proxy = Line2D([0], [0], color=color, linestyle='-', linewidth=1, label=label)
            legend_handles.append(proxy)
            legend_labels.append(label)
    
    # 添加组件项
    for handle, label in zip(unique_component_plots, unique_component_labels):
        legend_handles.append(handle)
        # 如果需要格式化标签，使用r"$" + label + r"$"，否则直接使用label
        formatted_label = label
        # 如果标签看起来像数学公式，可以添加$符号
        if any(c in label for c in ['\\', '^', '_', '{', '}']):
            label_fixed = label.replace('\\\\', '\\')
            formatted_label = fr"${label_fixed}$"
        legend_labels.append(formatted_label)
    
    # 创建图例
    if ax_legend is not None:
        ax_legend.axis('off')
        # 使用自定义的handler_map来处理error bar图例
        from matplotlib.legend_handler import HandlerBase
        handler_map = {}
        for i, label in enumerate(legend_labels):
            if label == 'Data':
                # 为Data项指定自定义handler
                handler_map[legend_handles[i]] = ErrorBarHandler()
        
        ax_legend.legend(legend_handles, legend_labels, 
                        loc='center', 
                        fontsize=7,
                        frameon=False,
                        fancybox=True,
                        shadow=False,
                        borderpad=0.5,
                        labelspacing=0.5,
                        handlelength=2.0,
                        handletextpad=0.5,
                        columnspacing=1.0,
                        handleheight=0.7,
                        handler_map=handler_map)
    
def plot_dalitz_histograms(dalitz_data_list, pdf_pages):
    """绘制所有TH2F直方图，每个dalitz目录一张图（三列：Data, Fit, Pull）"""
    if not dalitz_data_list:
        return

    n_cols = 3  # 固定3列：Data, Fit, Pull

    for dir_name, histograms, xlabel, ylabel in dalitz_data_list:
        # 确保我们有需要的直方图
        if not all(key in histograms for key in ['hdata', 'hfit']):
            print(f"警告: 目录 {dir_name} 缺少必需的直方图 (hdata, hfit)")
            continue

        # 获取直方图
        data_hist = histograms['hdata']
        fit_hist = histograms['hfit']
        bkg_hist = histograms.get('hbkg', None)  # hbkg可选

        # 获取直方图数据
        data_values, x_edges, y_edges = data_hist.to_numpy()
        fit_values, _, _ = fit_hist.to_numpy()
        if bkg_hist is not None:
            bkg_values, _, _ = bkg_hist.to_numpy()
        else:
            bkg_values = np.zeros_like(data_values)

        # 计算pull分布: (data - fit) / sqrt(fit)
        threshold = 1e-10
        pull_values = np.zeros_like(data_values)
        mask = fit_values > threshold
        pull_values[mask] = (data_values[mask] - fit_values[mask]) / np.sqrt(fit_values[mask])

        x_min, x_max = x_edges[0], x_edges[-1]
        y_min, y_max = y_edges[0], y_edges[-1]

        # 计算Data和Fit的颜色范围
        data_max = np.max(data_values)
        fit_max = np.max(fit_values)
        vmax = max(data_max, fit_max)

        # 为当前目录创建一张独立的图
        fig_dalitz = plt.figure(figsize=(12, 3))
        gs = gridspec.GridSpec(1, n_cols, figure=fig_dalitz,
                              hspace=0.25, wspace=0.25,
                              top=0.92, bottom=0.12, left=0.08, right=0.92)

        # 添加总标题
        # fig_dalitz.suptitle(f"Dalitz Plot: {dir_name}", fontsize=14, fontweight='bold')

        # 第一列: Data
        ax_data = fig_dalitz.add_subplot(gs[0, 0])
        masked_data = np.ma.masked_where(data_values.T <= threshold, data_values.T)
        im_data = ax_data.imshow(masked_data,
                                 extent=[x_min, x_max, y_min, y_max],
                                 cmap='rainbow',
                                 aspect='auto',
                                 origin='lower',
                                 interpolation='nearest',
                                 vmax=vmax,
                                 zorder=1)
        ax_data.set_title("Data", fontsize=10, fontweight='bold')
        ax_data.set_xlabel(f"${xlabel}$", fontsize=12)
        ax_data.set_ylabel(f"${ylabel}$", fontsize=12)
        ax_data.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax_data.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax_data.tick_params(axis='both', which='both', direction='in', labelsize=10, left=True, bottom=True)
        ax_data.grid(True, alpha=0.6, zorder=0)
        ax_data.set_axisbelow(True)
        for spine in ax_data.spines.values():
            spine.set_edgecolor('gray')
            spine.set_alpha(0.6)
        # plt.colorbar(im_data, ax=ax_data, fraction=0.046, pad=0.04)

        # 第二列: Fit
        ax_fit = fig_dalitz.add_subplot(gs[0, 1])
        masked_fit = np.ma.masked_where(fit_values.T <= threshold, fit_values.T)
        im_fit = ax_fit.imshow(masked_fit,
                               extent=[x_min, x_max, y_min, y_max],
                               cmap='rainbow',
                               aspect='auto',
                               origin='lower',
                               interpolation='nearest',
                               vmax=vmax,
                               zorder=1)
        ax_fit.set_title("Fit", fontsize=10, fontweight='bold')
        ax_fit.set_xlabel(f"${xlabel}$", fontsize=12)
        ax_fit.set_ylabel(f"${ylabel}$", fontsize=12)
        ax_fit.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax_fit.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax_fit.tick_params(axis='both', which='both', direction='in', labelsize=10, left=True, bottom=True)
        ax_fit.grid(True, alpha=0.6, zorder=0)
        ax_fit.set_axisbelow(True)
        for spine in ax_fit.spines.values():
            spine.set_edgecolor('gray')
            spine.set_alpha(0.6)
        # plt.colorbar(im_fit, ax=ax_fit, fraction=0.046, pad=0.04)

        # 第三列: Pull
        ax_pull = fig_dalitz.add_subplot(gs[0, 2])
        # pull_absmax = max(np.nanmax(np.abs(pull_values)), 3)
        # 计算pull值的范围，对称设置
        pull_min = np.abs(np.min(pull_values)) if np.min(pull_values) < np.max(pull_values) else np.max(pull_values)
        pull_vmax = max(pull_min, 3)  # 至少显示到±3σ

        pull_cmap = plt.cm.RdBu_r
        im_pull = ax_pull.imshow(pull_values.T,
                                 extent=[x_min, x_max, y_min, y_max],
                                 cmap=pull_cmap,
                                 aspect='auto',
                                 origin='lower',
                                 interpolation='nearest',
                                 vmin=-pull_vmax, vmax=pull_vmax,
                                 zorder=1)
        ax_pull.set_title("Pull", fontsize=10, fontweight='bold')
        ax_pull.set_xlabel(f"${xlabel}$", fontsize=12)
        ax_pull.set_ylabel(f"${ylabel}$", fontsize=12)
        ax_pull.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax_pull.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax_pull.tick_params(axis='both', which='both', direction='in', labelsize=10, left=True, bottom=True)
        ax_pull.grid(True, alpha=0.6, zorder=0)
        for spine in ax_pull.spines.values():
            spine.set_edgecolor('gray')
            spine.set_alpha(0.6)
        # plt.colorbar(im_pull, ax=ax_pull, fraction=0.046, pad=0.04)

        # 每个目录单独保存为一页
        pdf_pages.savefig(fig_dalitz, bbox_inches='tight')
        plt.close(fig_dalitz)

def plot_combined_histogram_with_pull(ax_top, ax_bottom, histograms, dir_name, xlabel, ylabel, legend_list=None):
    """在上下两个轴上绘制直方图和pull分布，不再单独绘制hfit，而是绘制hfit+hbkg"""
    from matplotlib.ticker import ScalarFormatter

    # 定义线型列表，与create_legend_column保持一致
    line_styles_list = ['-', '--', '-.', ':']

    # 首先在上方轴绘制直方图
    # 分离不同类型的直方图
    resonance_hists = []
    main_hists = []

    for name, hist in histograms.items():
        if name in ['hdata', 'hfit', 'hbkg']:
            main_hists.append((name, hist))
        else:
            resonance_hists.append((name, hist))

    # 为共振态分量准备颜色和线型
    num_resonances = len(resonance_hists)
    if num_resonances > 0:
        # 如果超过20种颜色，使用颜色+线型组合
        if num_resonances <= 20:
            # 使用tab20颜色映射
            colors = plt.cm.tab20(np.linspace(0, 1, num_resonances))
            linestyles = ['-'] * num_resonances  # 所有使用实线
        else:
            # 超过20种，循环使用颜色和线型
            # 使用tab20颜色，循环使用
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = []

            for i in range(num_resonances):
                # 循环使用颜色
                color_idx = i % 20
                # 每20种颜色换一种线型
                linestyle_idx = (i // 20) % len(line_styles_list)
                colors.append(base_colors[color_idx])
            
        # 计算所有直方图的最小和最大值以设置合适的y轴范围
        y_min = None
        y_max = None
        
        for name, hist in histograms.items():
            # 更新y轴范围
            values = hist.to_numpy()[0]
            # 忽略零值或负值（可能由于统计误差）
            positive_values = values[values > 0]
            if len(positive_values) > 0:
                hist_min = np.min(positive_values)
                hist_max = np.max(values)  # 使用所有值，包括零
                if y_min is None or hist_min < y_min:
                    y_min = hist_min
                if y_max is None or hist_max > y_max:
                    y_max = hist_max

        # 如果没找到正值，使用默认范围
        if y_min is None or y_max is None:
            y_min = 0
            y_max = 1

        component_plots = []
        component_labels = []

        # 绘制主要直方图（注意顺序：先绘制背景，确保它在最底层）
        main_plots = []
        
        # 获取背景直方图（如果存在）
        background_hist = None
        fit_hist = None
        data_hist = None
        
        for name, hist in main_hists:
            if name == 'hbkg':
                background_hist = hist
            elif name == 'hfit':
                fit_hist = hist
            elif name == 'hdata':
                data_hist = hist
        
        # 首先绘制背景（灰色填充区域）
        background_patch = None
        if background_hist is not None:
            edges = background_hist.to_numpy()[1]
            values = background_hist.to_numpy()[0]
            
            # 创建阶梯图的x和y
            step_centers = np.zeros(len(values) * 2 + 2)
            step_values = np.zeros(len(values) * 2 + 2)
            
            for i in range(len(values)):
                # 每个bin的左右边界
                step_centers[2*i] = edges[i]
                step_centers[2*i+1] = edges[i+1]
                step_values[2*i] = values[i]
                step_values[2*i+1] = values[i]
            
            # 添加最后的点
            step_centers[-2] = edges[-1]
            step_centers[-1] = edges[-1]
            step_values[-2] = values[-1] if len(values) > 0 else 0
            step_values[-1] = 0
            
            # 绘制阶梯图并填充下方区域
            background_line, = ax_top.plot(step_centers, step_values, 
                                          color='gray', linewidth=1, alpha=0.7,
                                          label='Background')
            
            # 填充阶梯图下方区域
            background_fill = ax_top.fill_between(step_centers, 0, step_values,
                                                 color='gray', alpha=0.3,
                                                 edgecolor='none')
            
            background_patch = (background_line, background_fill)
            main_plots.append(background_line)
        
        # 然后绘制共振态分量（使用阶梯图）
        for i, (name, hist) in enumerate(resonance_hists):
            edges = hist.to_numpy()[1]
            values = hist.to_numpy()[0]

            # 获取图例标签
            if legend_list is not None and len(legend_list) > 0 and i < len(legend_list):
                component_label = legend_list[i]
            else:
                # 格式化直方图名称作为图例
                component_label = name.replace('h_', '').replace('-', ' ').replace('_', ' ')

            # 确定线型
            if num_resonances <= 20:
                linestyle = '-'
            else:
                # 超过20种，使用不同的线型
                linestyle = line_styles_list[(i // 20) % len(line_styles_list)]

            # 创建阶梯图的x和y
            step_centers = np.zeros(len(values) * 2 + 2)
            step_values = np.zeros(len(values) * 2 + 2)
            
            for j in range(len(values)):
                # 每个bin的左右边界
                step_centers[2*j] = edges[j]
                step_centers[2*j+1] = edges[j+1]
                step_values[2*j] = values[j]
                step_values[2*j+1] = values[j]
            
            # 添加最后的点
            step_centers[-2] = edges[-1]
            step_centers[-1] = edges[-1]
            step_values[-2] = values[-1] if len(values) > 0 else 0
            step_values[-1] = 0

            line, = ax_top.plot(step_centers, step_values, 
                               linewidth=1, alpha=0.7,
                               color=colors[i], linestyle=linestyle, 
                               label=component_label)
            component_plots.append(line)
            component_labels.append(component_label)
        
        # 绘制数据点
        if data_hist is not None:
            edges = data_hist.to_numpy()[1]
            values = data_hist.to_numpy()[0]
            centers = (edges[:-1] + edges[1:]) / 2
            errors = np.sqrt(values)
            
            line = ax_top.errorbar(centers, values, yerr=errors,
                                   linestyle='none', marker='o', markersize=1, 
                                   color='black', label='Data',
                                   elinewidth=0.5, capsize=0, ecolor='black', linewidth=0)
            main_plots.append(line[0])
        
        # 绘制拟合+背景总和
        if fit_hist is not None:
            edges = fit_hist.to_numpy()[1]
            fit_values = fit_hist.to_numpy()[0]
            
            # 如果有背景，将背景加到拟合值中
            if background_hist is not None:
                background_values = background_hist.to_numpy()[0]
                total_values = fit_values + background_values
                label = 'Fit+Bkg'
            else:
                total_values = fit_values
                label = 'Fit'
            
            # 创建阶梯图的x和y
            step_centers = np.zeros(len(total_values) * 2 + 2)
            step_total_values = np.zeros(len(total_values) * 2 + 2)
            
            for i in range(len(total_values)):
                # 每个bin的左右边界
                step_centers[2*i] = edges[i]
                step_centers[2*i+1] = edges[i+1]
                step_total_values[2*i] = total_values[i]
                step_total_values[2*i+1] = total_values[i]
            
            # 添加最后的点
            step_centers[-2] = edges[-1]
            step_centers[-1] = edges[-1]
            step_total_values[-2] = total_values[-1] if len(total_values) > 0 else 0
            step_total_values[-1] = 0
            
            # 绘制阶梯图
            line, = ax_top.plot(step_centers, step_total_values, 
                               color='red', linewidth=1, alpha=1.0,
                               label=label)
            main_plots.append(line)

        # 设置上方轴的标签和样式
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))  # 根据你的数据范围调整
        ax_top.yaxis.set_major_formatter(formatter)
        ax_top.set_ylabel(f"${ylabel}$", fontsize=12)
        ax_top.grid(True, alpha=0.6)
        ax_top.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax_top.tick_params(axis='both', direction='in', which='both', labelsize=10, left=False, top=True, bottom=False)

        # 设置合理的y轴范围
        if y_max > y_min:
            y_margin = (y_max - y_min) * 0.1
            ax_top.set_ylim(max(0, y_min - y_margin), y_max + y_margin)

        # 移除上方轴的x轴标签，因为共用下方的x轴
        ax_top.set_xlabel("")
        ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        for spine in ax_top.spines.values():
            spine.set_edgecolor('gray')
            spine.set_alpha(0.6)

        # 现在在下方轴绘制pull分布
        if data_hist is None or fit_hist is None:
            ax_bottom.text(0.5, 0.5, 'No data/fit', ha='center', va='center')
            ax_bottom.set_xlabel(f"${xlabel}$", fontsize=8)
            # 设置x轴范围（从第一个直方图获取）
            if histograms:
                first_hist = next(iter(histograms.values()))
                edges = first_hist.to_numpy()[1]
                x_min = edges[0]
                x_max = edges[-1]
                ax_top.set_xlim(x_min, x_max)
                ax_bottom.set_xlim(x_min, x_max)
            return component_plots, component_labels, None, None

        # 获取数据值和拟合值
        data_values = data_hist.to_numpy()[0]
        fit_values = fit_hist.to_numpy()[0]
        edges = data_hist.to_numpy()[1]
        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]

        # 计算pull: (data - (fit+bkg)) / sqrt(fit+bkg)
        # 如果有背景，将背景加到拟合值中
        if background_hist is not None:
            background_values = background_hist.to_numpy()[0]
            total_fit_values = fit_values + background_values
        else:
            total_fit_values = fit_values
        
        # 避免除以零
        mask = total_fit_values > 0
        if not np.any(mask):
            ax_bottom.text(0.5, 0.5, 'No positive fit values', ha='center', va='center')
            ax_bottom.set_xlabel(f"${xlabel}$", fontsize=8)
            # 设置x轴范围
            x_min = edges[0]
            x_max = edges[-1]
            ax_top.set_xlim(x_min, x_max)
            ax_bottom.set_xlim(x_min, x_max)
            return component_plots, component_labels, None, None

        pull = np.zeros_like(data_values)
        pull[mask] = (data_values[mask] - total_fit_values[mask]) / np.sqrt(total_fit_values[mask])

        # 计算chi2和ndf
        chi2 = np.sum((data_values[mask] - total_fit_values[mask])**2 / total_fit_values[mask])
        ndf = np.sum(mask)  # 有效bin数
        reduced_chi2 = chi2 / ndf if ndf > 0 else 0

        # 绘制pull分布
        ax_bottom.bar(centers[mask], pull[mask], width=widths[mask],
                      linewidth=0.5, facecolor='gray', alpha=0.9)

        # 添加参考线 y=0
        ax_bottom.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        # 添加±3σ线
        ax_bottom.axhline(y=3, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_bottom.axhline(y=-3, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

        # 设置下方轴的标签和样式
        ax_bottom.set_xlabel(f"${xlabel}$", fontsize=12)
        ax_bottom.set_ylabel('Pull', fontsize=12)
        ax_bottom.grid(True, alpha=0.6)
        ax_bottom.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax_bottom.tick_params(axis='both', which='both', labelsize=10, direction='in', left=True, bottom=True)

        for spine in ax_bottom.spines.values():
            spine.set_edgecolor('gray')
            spine.set_alpha(0.6)

        # 设置y轴范围，留出一些边距
        pull_range = max(abs(pull[mask])) if np.any(mask) else 1
        if pull_range > 5:
            ax_bottom.set_ylim(-pull_range * 1.2, pull_range * 1.2)
        else:
            ax_bottom.set_ylim(-6, 6)

        # 在上方轴的左上角显示chi2/ndf
        text_str = f'$\\chi^2$/ndf = {reduced_chi2:.2f}'
        ax_bottom.text(0.05, 0.95, text_str, transform=ax_bottom.transAxes,
                fontsize=6, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))

        # 确保上下子图的x轴范围相同
        x_min = edges[0]
        x_max = edges[-1]
        ax_top.set_xlim(x_min, x_max)
        ax_bottom.set_xlim(x_min, x_max)

        return component_plots, component_labels, chi2, ndf
    else:
        # 如果没有共振态分量
        return [], [], None, None

def plot_nll_history(run_id):
    """绘制指定run的NLL历史"""
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np

    nll_file = "results/nll_history.txt"
    if not os.path.exists(nll_file):
        print(f"NLL历史文件不存在: {nll_file}")
        return

    # 读取NLL历史数据
    run_data = []
    with open(nll_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                current_run = int(parts[0])
                iteration = int(parts[1])
                nll = float(parts[2])
                if current_run == run_id:
                    run_data.append((iteration, nll))

    if not run_data:
        print(f"未找到第 {run_id} 次运行的NLL历史数据")
        return

    iterations, nll_values = zip(*run_data)

    # 创建图形
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, nll_values, 'b-', linewidth=2, label=f'Run {run_id}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('NLL', fontsize=12)
    plt.title(f'NLL History for Run {run_id}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存图形
    output_file = f"results/nll_history_run_{run_id}.pdf"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"NLL历史图已保存到: {output_file}")

def generate_weight_file_from_params(run_id):
    """从optimized_parameters.txt读取指定run的参数，生成权重文件"""
    import ctpwa
    import numpy as np
    import torch

    output_file = f"results/weight_run_{run_id}.root"
    if output_file is not None and os.path.exists(output_file):
        print(f"权重文件已存在: {output_file}")
        return output_file

    params_file = "results/optimized_parameters.txt"
    if not os.path.exists(params_file):
        print(f"参数文件不存在: {params_file}")
        return "results/weight_best.root"

    # 读取参数文件，提取指定run的参数
    params_list = []
    in_target_run = False
    amplitudes_count = 0

    with open(params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 7:
                current_run = int(parts[0])

                # 检查是否进入目标run
                if current_run == run_id:
                    in_target_run = True
                elif in_target_run:
                    # 已经离开目标run，停止读取
                    break

                if in_target_run:
                    # 解析参数：RealPart和ImagPart
                    real_part = float(parts[3])
                    imag_part = float(parts[4])
                    params_list.append(complex(real_part, imag_part))
                    amplitudes_count += 1

    if not params_list:
        print(f"未找到第 {run_id} 次运行的参数")
        return "results/weight_best.root"

    print(f"找到第 {run_id} 次运行的 {len(params_list)} 个参数")

    # 创建ctpwa分析对象
    try:
        ana = ctpwa.analysis()

        # 将参数转换为torch张量
        params_tensor = torch.tensor(params_list, dtype=torch.complex64, device='cuda')

        # 生成权重文件
        ana.writeWeightFile(params_tensor, output_file, 0)
        print(f"权重文件已生成: {output_file}")
        return output_file
    except Exception as e:
        print(f"生成权重文件时出错: {e}")
        return "results/weight_best.root"

def main(args=None):
    # 解析命令行参数
    if args is None:
        parser = argparse.ArgumentParser(description='PWA结果绘图工具')
        parser.add_argument('--params', type=int, help='使用第x次run的参数生成权重文件并画图')
        parser.add_argument('--nll', type=int, help='画第x次run的似然值分布')
        args = parser.parse_args()

    # 如果指定了--nll参数，绘制NLL历史
    if args.nll is not None:
        plot_nll_history(args.nll)
        return

    # 如果指定了--params参数，使用指定run的参数生成权重文件
    if args.params is not None:
        weight_file = generate_weight_file_from_params(args.params)
    else:
        # 默认使用最佳权重文件
        weight_file = "results/weight_best.root"

    # 打开ROOT文件
    file_path = weight_file
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return
    
    try:
        file = uproot.open(file_path)
    except Exception as e:
        print(f"打开文件时出错: {e}")
        return
    
    # 读取图例
    legend_list = read_legends(file)
    
    # 分类存储目录
    mass_dirs = []
    cosbeta_dirs = []
    h1f_dirs = []
    dalitz_dirs = []
    
    # 遍历文件中的目录
    for key in file.keys():
        if isinstance(file[key], uproot.ReadOnlyDirectory):
            dir_name = key.split(';')[0]
            if 'mass' in dir_name:
                mass_dirs.append(dir_name)
                h1f_dirs.append(dir_name)
            elif 'cosbeta' in dir_name:
                cosbeta_dirs.append(dir_name)
                h1f_dirs.append(dir_name)
            elif 'dalitz' in dir_name:
                dalitz_dirs.append(dir_name)
    
    print(f"共 {len(mass_dirs)} 个mass目录")
    print(f"共 {len(cosbeta_dirs)} 个cosbeta目录")
    print(f"共 {len(dalitz_dirs)} 个dalitz目录")
    
    # 准备PDF输出文件
    if args.params is not None:
        pdf_path = f"results/results_plot_run_{args.params}.pdf"
    else:
        pdf_path = "results/results_plot.pdf"
    with PdfPages(pdf_path) as pdf_pages:
        
        # 第一页：所有h1f直方图（质量和cosbeta）与pull分布
        if h1f_dirs:
            n_plots = len(h1f_dirs)  # 每个目录需要一个组合图

            # 确定布局：尝试使布局接近正方形
            n_rows, n_cols = determine_subplot_layout(n_plots)

            # 创建图形，留出右侧空间用于图例
            fig_h1f = plt.figure(figsize=(12, 8))  # A4横向
            gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 0.1], wspace=0.05)

            # 左侧：子图区域
            gs_left = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                      subplot_spec=gs[0],
                                                      hspace=0.4, wspace=0.3)

            # 右侧：图例区域
            gs_right = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                       subplot_spec=gs[1])

            all_component_plots = []
            all_component_labels = []
            has_background = False

            # 绘制所有组合图（直方图 + pull分布）
            for i, dir_name in enumerate(h1f_dirs):
                if i >= n_rows * n_cols:
                    break

                row = i // n_cols
                col = i % n_cols

                # 在每个网格位置创建两个垂直排列的子图
                # 使用subgridspec创建两个子图，高度比例约为3:1
                sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                         subplot_spec=gs_left[row, col],
                                                         height_ratios=[3, 1], hspace=0.05)

                # 创建上方轴（直方图）
                ax_top = fig_h1f.add_subplot(sub_gs[0])
                # 创建下方轴（pull分布）
                ax_bottom = fig_h1f.add_subplot(sub_gs[1])

                dir_obj = file[dir_name]

                # 获取标签
                xlabel, ylabel = get_labels(dir_obj)

                # 获取所有TH1F直方图
                histograms = get_th1f_histograms(dir_obj)

                # 检测是否有h_bkg
                if 'hbkg' in histograms:
                    has_background = True

                # 绘制组合图
                component_plots, component_labels, chi2, ndf = plot_combined_histogram_with_pull(
                    ax_top, ax_bottom, histograms, dir_name, xlabel, ylabel, legend_list)

                all_component_plots.append(component_plots)
                all_component_labels.append(component_labels)

            # 隐藏多余的网格位置
            for i in range(len(h1f_dirs), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                ax = fig_h1f.add_subplot(gs_left[row, col])
                ax.axis('off')

            # 右侧：创建图例列
            ax_legend = fig_h1f.add_subplot(gs_right[0])
            create_legend_column(fig_h1f, all_component_plots, all_component_labels, legend_list, ax_legend, has_background)

            # 保存到PDF第一页
            pdf_pages.savefig(fig_h1f, bbox_inches='tight')
            plt.close(fig_h1f)
        
        # Dalitz图（每个目录单独一页）
        if dalitz_dirs:
            # 收集所有dalitz直方图及其标签
            dalitz_data_list = []
            for dir_name in dalitz_dirs:
                dir_obj = file[dir_name]
                
                # 获取dalitz图的标签
                xlabel, ylabel = get_labels(dir_obj)
                
                # 获取所有TH2F直方图
                histograms = get_th2f_histograms(dir_obj)
                if histograms:
                    dalitz_data_list.append((dir_name, histograms, xlabel, ylabel))
            
            # 绘制所有dalitz图到第二页
            plot_dalitz_histograms(dalitz_data_list, pdf_pages)

        # 保存PDF的元数据
        d = pdf_pages.infodict()
        d['Title'] = 'ROOT Histogram Analysis'
        d['Author'] = 'Analysis Script'
        d['Subject'] = 'Mass and Dalitz distributions'
        d['Keywords'] = 'ROOT, histograms, analysis'
    
    print(f"所有图形已保存到 {pdf_path}")
    
    # 关闭文件
    file.close()

if __name__ == "__main__":
    main()
