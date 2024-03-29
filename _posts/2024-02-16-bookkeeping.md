---
layout: post
title: "一个简单的记账应用实践"
date: 2024-02-16
excerpt: "一个简单的记账应用实践"
tags: [Agent, bookkeeping, bot]
comments: true
nlp: false 
---

# 💰快速记账功能的简单实践

## 效果展示

某人在小红书看到有个苹果手机敲两下背面就可以实现自动记账的功能，希望我也实现一个。研究了一下，使用苹果的快捷指令 + 飞书的多维表格，实现了这个功能。它的好处其实只是帮你实现了金额的自动提取（不管是支付宝、微信还是云闪付）和自动记录，省去打开记账APP、输入金额这一两步而已。不过自己这几天用下来，倒是也让我养成了花了钱就习惯敲两下记一下😹。

效果如下：

![1708077280783030 (1)]({{ site.url }}/assets/img/bookkeeping/1708077280783030 (1).gif)

飞书表格则是自动添加新的记录：

![image-20240216181151810]({{ site.url }}/assets/img/bookkeeping/image-20240216181151810.png)

结合多维表格的仪表盘，倒是可以实现非常丰富的各种图表。

![image-20240216181339959]({{ site.url }}/assets/img/bookkeeping/image-20240216181339959.png)

不过某人后面又要换vivo手机，所以后面还继续研究了安卓手机上怎么实现这块操作，总体来说对比苹果手机，安卓上体验要差挺多。在研究了coze和飞书机器人后，最终还是选择了飞书机器人会简单一些。

![20240216-185100]({{ site.url }}/assets/img/bookkeeping/20240216-185100.gif)


## 核心思路

核心其实很简单，OCR识别屏幕 -> 提取金额 -> 选择类别 -> 表格 -> 自动更新。分两大块：

- 表格的自动更新：这块使用飞书的多维表格，提供了非常多自动化能力，直接用就好了
- 自动化提取：OCR + webhook的触发

## 飞书自动记账表格

首先利用飞书多维表格建立一个支出表格（前面那张表）和自动化解析表格，使用函数功能实现自动化的信息抽取。

![image-20240216185539346]({{ site.url }}/assets/img/bookkeeping/image-20240216185539346.png)

构建自动化流程，使得每月和每笔支出都自动记录到表格里面。

<img src="{{ site.url }}/assets/img/bookkeeping/image-20240216194126578.png" alt="image-20240216194126578" style="zoom:50%;" />

每当记录变更，自动提取字段并填到支持表格。

![image-20240216193932200]({{ site.url }}/assets/img/bookkeeping/image-20240216193932200.png)

仪表盘就很简单了，自己配配就好。

最后配置一个飞书机器人，通过webhook触发更新自动提取表格的内容，从而实现了自动记账。

![image-20240216200520353]({{ site.url }}/assets/img/bookkeeping/image-20240216200520353.png)

## 手机上实现

苹果手机的快捷指令非常方便，提供了很多功能，其实这里我们需要的核心就是截屏+OCR+正则提取金额+调用webhook，直接用指令拼起来就好了，再加一些选取和筛选的操作，就实现前面的功能。

最后配置一下敲击手机背面两下触发这个快捷记账的指令，苹果手机上的配置就完成了。

![image-20240216200716518]({{ site.url }}/assets/img/bookkeeping/image-20240216200716518.png)

## 安卓机上实现

由于某人要换X100，这前面的配置也只能我自己用用了，继续研究安卓机上怎么操作。我拿着X100研究了半天也没发现有类似苹果的快捷指令那么细的自定义能力，基本都是配置一些简单的快捷方式。这块倒是有一些APP能够做快捷指令，但是没有深入研究，先看看能不能用现有的能力做一些什么。

目前想到的方案就是，先通过截图，然后分享到某个APP（如飞书或者微信），再自动提取金额，然后选择分类和保存。这里选择了飞书，主要是一开始用coze的时候可以直接部署到飞书上。

### Coze - Agent的方法

Coze是字节出的类似GPTs的平台，比较方便用于搭建Agent，而且能够一键部署到飞书，非常方便。因此看看能不能用它实现个简单的记账的Agent。在coze插件里面有OCR能力，那就直接撸一个。

![image-20240216202141480]({{ site.url }}/assets/img/bookkeeping/image-20240216202141480.png)

通过自然语言的描述安排好整体的工作流程，再额外配置两个工作流：

- extract_money: 用于识别图片中的OCR信息，并提取出金额列表（截图里面可能包含多个金额）

  ![image-20240216202416297]({{ site.url }}/assets/img/bookkeeping/image-20240216202416297.png)

- Save: 将金额和分类，以及备注保存到飞书表格中

  ![image-20240216202444756]({{ site.url }}/assets/img/bookkeeping/image-20240216202444756.png)

原来一直不清除工作流到底有什么意义，现在看来其实就是提供了一个方法让用户自己快速实现自己的function。效果如下：

<img src="{{ site.url }}/assets/img/bookkeeping/image-20240216203853437.png" alt="image-20240216203853437" style="zoom:30%;" />

但是，体验还很差，主要以下几块原因：

- 大模型太慢了，我传个图片，要等个十几二十秒才能帮我识别金额，其他聊天也需要等待很久才能给我回复和保存。有这时间，我不如自己去文档里面填内容了。

- 幻觉，不稳定的情况很多，比如莫名其妙出个没有的金额和时间；不智能，多次交互才能帮你保存。

  <img src="{{ site.url }}/assets/img/bookkeeping/image-20240216204105604.png" alt="image-20240216204105604" style="zoom:33%;" />

<img src="{{ site.url }}/assets/img/bookkeeping/image-20240216204201627.png" alt="image-20240216204201627" style="zoom:33%;" />

体验多次之后，Agent并不适合这种需要高效的场景，简直是杀鸡用牛刀。

### 自己做个飞书机器人

没办法，只能选择自己撸个机器人。我这里不需要用对话做交互，只需要用户自己点选几个选项即可，飞书刚好有消息卡片感觉很适合这个场景。

飞书虽然提供了OCR的接口，但是老是调用不成功，于是转而用百度的OCR，还好有白嫖额度，每个月2000次，基本也够用了，速度也挺快。

于是对着飞书给的示例代码开始撸，但是研究了半天飞书的文档，发现它的卡片虽然不错，但是是每个控件都有独立的交互，并不是表单形式，这导致也无法实现我的需求。我需要用户选择完金额、类别后统一提交。

正当我准备放弃的时候，借用万能的google，居然让我翻到了还在内测的表单功能，赞，刚好满足我的需求。

<img src="{{ site.url }}/assets/img/bookkeeping/image-20240216205104072.png" alt="image-20240216205104072" style="zoom:33%;" />

对着样例终于实现了想要的功能。

<img src="{{ site.url }}/assets/img/bookkeeping/image-20240216205716465.png" alt="image-20240216205716465" style="zoom:33%;" />

这个方法体验下来：

- 速度会快很多，OCR也挺快，表单操作比对话交互要简单很多
- 步骤整体还是比苹果多，截屏 + 分享到飞书机器人 + 选择表单 + 提交，而苹果只需要双击背面，选择表单。只能说是曲线救国了。
- 说起来，还不如自己打开个记账APP，手动输入一下，时间其实差不多。

## 后记

其实记账应该是个非常小的功能，感觉要让人用的话主要还是要足够方便和简洁，能少点就少点，能少操作就少操作，别硬上Agent。

不过自己从头到尾撸了一个记账功能，能够定义符合自己需求的图表，算是给自己做了个功能吧，至少让自己每次消费了都会习惯性的记录一下（别浪费了过年这几天费的时间😂）。

