# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Label Conductor
# Copyright (c) 2017 by Contributors
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Shiqi
# --------------------------------------------------------

from easydict import EasyDict as edict
import os

HRSC_class_label = {
    '100000000': 'ship',
    '100100000': 'aircraft_carrier',
    '100100001': 'nimitz_class_aircraft_carrier',
    '100100002': 'enterprise_class_aircraft_carrier',
    '100100003': 'kitty_hawk_class_aircraft_carrier',
    '100100004': 'admiral_kuznetsov_aircraft_carrier',
    '100100005': 'tarawa_class_amphibious_assault_ship',
    '100100006': 'ford_class_aircraft_carriers',
    '100100007': 'midway_class_aircraft_carrier',
    '100100008': 'invincible_class_aircraft_carrier',
    '100200000': 'warcraft',
    '100200001': 'arleigh_burke_class_destroyers',
    '100200002': 'whidbey_island_class_landing_craft',
    '100200003': 'perry_class_frigate',
    '100200004': 'sanantonio_class_amphibious_transport_dock',
    '100200005': 'ticonderoga_class_cruiser',
    '100200006': 'abukuma_class_destroyer_escort',
    '100200007': 'austen_class_amphibious_transport_dock',
    '100200008': 'uss_blue_ridge_LCC_19',
    '100200009': 'commander_ship_round_head_ox_tail',
    '100200010': 'lute_style_warcraft',
    '100200011': 'medical_ship',
    '100300000': 'merchant_ship',
    '100300001': 'container_ship',
    '100300002': 'car_carrier_round',
    '100300003': 'hovercraft',
    '100300004': 'yacht',
    '100300005': 'cargo_ship',
    '100300006': 'cruise_ship',
    '100300007': 'car_carrier_one_side_flat',
    '100400000': 'submarine',
}


class LabelConductor:
    """
    目标检测标记文件处理器。支持VOC数据集的标记格式。
    """
    def __init__(self, path=''):
        # root path
        self.root_path = './data/VOCdevkit/VOC2007'
        if path:
            self.set_root_path(path)
        # data path
        self.anno_path = os.path.join(self.root_path, 'Annotations')
        self.img_path = os.path.join(self.root_path, 'JPEGImages')
        self.testset_path = os.path.join(self.root_path, 'ImageSets', 'Main', 'test.txt')
        self.trainset_path = os.path.join(self.root_path, 'ImageSets', 'Main', 'train.txt')
        self.valset_path = os.path.join(self.root_path, 'ImageSets', 'Main', 'val.txt')
        self.trainvalset_path = os.path.join(self.root_path, 'ImageSets', 'Main', 'trainval.txt')

        # voc annotation structure
        self.voc = edict()
        self.voc.folder = ''
        self.voc.filename = ''
        self.voc.segmented = ''

        self.voc.source = edict()
        self.voc.source.database = ''
        self.voc.source.annotation = ''
        self.voc.source.image = ''
        self.voc.source.flickrid = ''

        self.voc.owner = edict()
        self.voc.owner.flickrid = ''
        self.voc.owner.name = ''

        self.voc.size = edict()
        self.voc.size.width = ''
        self.voc.size.height = ''
        self.voc.size.depth = ''

        # objects包含了一幅图像中的所有object标签
        self.voc.objects = []
        '''
        # 一个object标签定义。使用append_object将标签加入objects列表
        self.voc_object = edict()
        self.voc_object.name = ''
        self.voc_object.pose = ''
        self.voc_object.truncated = ''
        self.voc_object.difficult = ''
        self.voc_object.bndbox = edict()
        self.voc_object.bndbox.xmin = ''
        self.voc_object.bndbox.ymin = ''
        self.voc_object.bndbox.xmax = ''
        self.voc_object.bndbox.ymax = ''
        '''

    def set_root_path(self, path):
        """
        修改数据集root路径（./path/to/VOC20XX）
        :param path: 新的root路径
        :return: None
        """
        self.root_path = path

    '''
    def append_object(self):
        """
        将当前的voc_object加入objects列表
        :return: None
        """
        self.voc.objects.append(self.voc_object)
    
    def clear_temp_object(self):
        """
        清空voc_object
        :return: None
        """
        self.voc_object.name = ''
        self.voc_object.pose = ''
        self.voc_object.truncated = ''
        self.voc_object.difficult = ''
        self.voc_object.bndbox.xmin = ''
        self.voc_object.bndbox.ymin = ''
        self.voc_object.bndbox.xmax = ''
        self.voc_object.bndbox.ymax = ''
    '''

    def clear_objects(self):
        """
        清空objects列表
        :return: None
        """
        self.voc.objects = []

    def clear_all(self):
        """
        清空当前对象中缓存的所有voc数据
        :return: None
        """
        self.voc.folder = ''
        self.voc.filename = ''
        self.voc.segmented = ''
        self.voc.source.database = ''
        self.voc.source.annotation = ''
        self.voc.source.image = ''
        self.voc.source.flickrid = ''
        self.voc.owner.flickrid = ''
        self.voc.owner.name = ''
        self.voc.size.width = ''
        self.voc.size.height = ''
        self.voc.size.depth = ''
        self.voc.objects = []
        '''
        self.voc_object.name = ''
        self.voc_object.pose = ''
        self.voc_object.truncated = ''
        self.voc_object.difficult = ''
        self.voc_object.bndbox.xmin = ''
        self.voc_object.bndbox.ymin = ''
        self.voc_object.bndbox.xmax = ''
        self.voc_object.bndbox.ymax = ''
        '''

    def object_count(self):
        """
        返回当前对象中存储的目标个数
        :return: 目标个数
        """
        return len(self.voc.objects)

    def decompose(self, index, silence=True):
        """
        解析一个voc标注文件，将内容缓存到对象
        :param index: 标注文件编号
        :param silence: 控制是否隐藏解析过程的控制台输出
        :return: None
        """
        import xml.etree.ElementTree as ET

        if isinstance(index, int):
            index = str(index)
        file_path = os.path.join(self.anno_path, index + '.xml')
        if os.path.isfile(file_path):
            tree = ET.parse(file_path)
        else:
            print 'Path \'{}\' is invalid.'.format(file_path)
            return
        # 清空缓存内容，防止新文件信息不全导致写入旧文件信息
        self.clear_all()
        # 获取xml内容
        self.voc.folder = tree.find('folder').text if tree.find('folder') is not None else ''
        self.voc.filename = tree.find('filename').text if tree.find('filename') is not None else ''
        self.voc.segmented = tree.find('segmented').text if tree.find('segmented') is not None else ''
        source = tree.find('source')
        if source is not None:
            self.voc.source.database = source.find('database').text if source.find('database') is not None else ''
            self.voc.source.annotation = source.find('annotation').text if source.find('annotation') is not None else ''
            self.voc.source.image = source.find('image').text if source.find('image') is not None else ''
            self.voc.source.flickrid = source.find('flickrid').text if source.find('flickrid') is not None else ''
        owner = tree.find('owner')
        if owner is not None:
            self.voc.owner.flickrid = owner.find('flickrid').text if owner.find('flickrid') is not None else ''
            self.voc.owner.name = owner.find('name').text if owner.find('name') is not None else ''
        size = tree.find('size')
        if size is not None:
            self.voc.size.width = size.find('width').text if size.find('width') is not None else ''
            self.voc.size.height = size.find('height').text if size.find('height') is not None else ''
            self.voc.size.depth = size.find('depth').text if size.find('depth') is not None else ''

        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            # 清空缓存内容，防止新文件信息不全导致写入旧文件信息
            # self.clear_temp_object()

            # 定义object标签（用于python list append，必须在循环内定义）
            voc_object = edict()
            voc_object.name = ''
            voc_object.pose = ''
            voc_object.truncated = ''
            voc_object.difficult = ''
            voc_object.bndbox = edict()
            voc_object.bndbox.xmin = ''
            voc_object.bndbox.ymin = ''
            voc_object.bndbox.xmax = ''
            voc_object.bndbox.ymax = ''

            if obj.find('name') is not None:
                voc_object.name = obj.find('name').text
            else:
                print 'can\'t find attribute \'name\', object discarded.'
                continue
            voc_object.pose = obj.find('pose').text if obj.find('pose') is not None else ''
            voc_object.truncated = obj.find('truncated').text if obj.find('truncated') is not None else ''
            voc_object.difficult = obj.find('difficult').text if obj.find('difficult') is not None else ''
            bbox = obj.find('bndbox')
            if bbox is not None:
                if bbox.find('xmin') is not None:
                    voc_object.bndbox.xmin = bbox.find('xmin').text
                else:
                    print 'can\'t find attribute \'xmin\', object discarded.'
                    continue
                if bbox.find('ymin') is not None:
                    voc_object.bndbox.ymin = bbox.find('ymin').text
                else:
                    print 'can\'t find attribute \'ymin\', object discarded.'
                    continue
                if bbox.find('xmax') is not None:
                    voc_object.bndbox.xmax = bbox.find('xmax').text
                else:
                    print 'can\'t find attribute \'xmax\', object discarded.'
                    continue
                if bbox.find('ymax') is not None:
                    voc_object.bndbox.ymax = bbox.find('ymax').text
                else:
                    print 'can\'t find attribute \'ymax\', object discarded.'
                    continue
            else:
                print 'can\'t find bounding box, object discarded.'
                continue
            # 将解析出的object添加到object列表中
            # self.append_object()
            self.voc.objects.append(voc_object)
        if not silence:
            print '\'{}.xml\' proceeded, found {} objects.'.format(index, self.object_count())

    def load_objects(self, index, silence=True):
        """
        解析一个voc标注文件中的object标签（不写入对象的objects列表）
        :param index: 标注文件编号
        :param silence: 控制是否隐藏load过程的控制台输出
        :return: object列表
        """
        import xml.etree.ElementTree as ET

        if isinstance(index, int):
            index = str(index)
        file_path = os.path.join(self.anno_path, index + '.xml')
        if os.path.isfile(file_path):
            tree = ET.parse(file_path)
        else:
            print 'Path \'{}\' is invalid.'.format(file_path)
            return
        objects = []
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            # 清空缓存内容，防止新文件信息不全导致写入旧文件信息
            # self.clear_temp_object()

            # 定义object标签（用于python list append，必须在循环内定义）
            voc_object = edict()
            voc_object.name = ''
            voc_object.pose = ''
            voc_object.truncated = ''
            voc_object.difficult = ''
            voc_object.bndbox = edict()
            voc_object.bndbox.xmin = ''
            voc_object.bndbox.ymin = ''
            voc_object.bndbox.xmax = ''
            voc_object.bndbox.ymax = ''

            if obj.find('name') is not None:
                voc_object.name = obj.find('name').text
            else:
                print 'can\'t find attribute \'name\', object discarded.'
                continue
            voc_object.pose = obj.find('pose').text if obj.find('pose') is not None else ''
            voc_object.truncated = obj.find('truncated').text if obj.find('truncated') is not None else ''
            voc_object.difficult = obj.find('difficult').text if obj.find('difficult') is not None else ''
            bbox = obj.find('bndbox')
            if bbox is not None:
                if bbox.find('xmin') is not None:
                    voc_object.bndbox.xmin = bbox.find('xmin').text
                else:
                    print 'can\'t find attribute \'xmin\', object discarded.'
                    continue
                if bbox.find('ymin') is not None:
                    voc_object.bndbox.ymin = bbox.find('ymin').text
                else:
                    print 'can\'t find attribute \'ymin\', object discarded.'
                    continue
                if bbox.find('xmax') is not None:
                    voc_object.bndbox.xmax = bbox.find('xmax').text
                else:
                    print 'can\'t find attribute \'xmax\', object discarded.'
                    continue
                if bbox.find('ymax') is not None:
                    voc_object.bndbox.ymax = bbox.find('ymax').text
                else:
                    print 'can\'t find attribute \'ymax\', object discarded.'
                    continue
            else:
                print 'can\'t find bounding box, object discarded.'
                continue
            objects.append(voc_object)
        if not silence:
            print '\'{}.xml\' proceeded, found {} objects.'.format(index, len(objects))
        return objects

    def print2bash(self):
        """
        将annotation文件内容输出到控制台
        :return: None
        """
        print '<annotation>'
        print '  <folder>{}</folder>'.format(self.voc.folder)
        print '  <filename>{}</filename>'.format(self.voc.filename)
        print '  <source>'
        print '    <database>{}</database>'.format(self.voc.source.database)
        print '    <annotation>{}</annotation>'.format(self.voc.source.annotation)
        print '    <image>{}</image>'.format(self.voc.source.image)
        print '    <flickrid>{}</flickrid>'.format(self.voc.source.flickrid)
        print '  </source>'
        print '  <owner>'
        print '    <flickrid>{}</flickrid>'.format(self.voc.owner.flickrid)
        print '    <name>{}</name>'.format(self.voc.owner.name)
        print '  </owner>'
        print '  <size>'
        print '    <width>{}</width>'.format(self.voc.size.width)
        print '    <height>{}</height>'.format(self.voc.size.height)
        print '    <depth>{}</depth>'.format(self.voc.size.depth)
        print '  </size>'
        print '  <segmented>{}</segmented>'.format(self.voc.segmented)
        if len(self.voc.objects) > 0:
            for voc_object in self.voc.objects:
                print '  <object>'
                print '    <name>{}</name>'.format(voc_object.name)
                print '    <pose>{}</pose>'.format(voc_object.pose)
                print '    <truncated>{}</truncated>'.format(voc_object.truncated)
                print '    <difficult>{}</difficult>'.format(voc_object.difficult)
                print '    <bndbox>'
                print '      <xmin>{}</xmin>'.format(voc_object.bndbox.xmin)
                print '      <ymin>{}</ymin>'.format(voc_object.bndbox.ymin)
                print '      <xmax>{}</xmax>'.format(voc_object.bndbox.xmax)
                print '      <ymax>{}</ymax>'.format(voc_object.bndbox.ymax)
                print '    </bndbox>'
                print '  </object>'
        print '</annotation>'

    def print2file(self, index, output_path=''):
        """
        将annotation文件内容输出到文件
        :param index: 文件名称（不带扩展名）
        :param output_path: 输出路径
        :return: None
        """
        if isinstance(index, int):
            index = str(index)
        # 组合路径
        if len(output_path):
            try:
                outf = open(os.path.join(output_path, index + '.xml'), 'w')
            except IOError:
                print 'unable to open file \'{}\'.'.format(os.path.join(output_path, index + '.xml'))
                return
            else:
                print '\'{}.xml\' found:'.format(index + '.xml')
        else:
            try:
                outf = open(index + '.xml', 'w')
            except IOError:
                print 'unable to open file \'{}\'.'.format(index + '.xml')
                return
            else:
                print '\'{}.xml\' found:'.format(index + '.xml')
        # 输出到文件
        outf.write('<annotation>')
        outf.write('  <folder>{}</folder>'.format(self.voc.folder))
        outf.write('  <filename>{}</filename>'.format(self.voc.filename))
        outf.write('  <source>')
        outf.write('    <database>{}</database>'.format(self.voc.source.database))
        outf.write('    <annotation>{}</annotation>'.format(self.voc.source.annotation))
        outf.write('    <image>{}</image>'.format(self.voc.source.image))
        outf.write('    <flickrid>{}</flickrid>'.format(self.voc.source.flickrid))
        outf.write('  </source>')
        outf.write('  <owner>')
        outf.write('    <flickrid>{}</flickrid>'.format(self.voc.owner.flickrid))
        outf.write('    <name>{}</name>'.format(self.voc.owner.name))
        outf.write('  </owner>')
        outf.write('  <size>')
        outf.write('    <width>{}</width>'.format(self.voc.size.width))
        outf.write('    <height>{}</height>'.format(self.voc.size.height))
        outf.write('    <depth>{}</depth>'.format(self.voc.size.depth))
        outf.write('  </size>')
        outf.write('  <segmented>{}</segmented>'.format(self.voc.segmented))
        if len(self.voc.objects) > 0:
            for voc_object in self.voc.objects:
                outf.write('  <object>')
                outf.write('    <name>{}</name>'.format(voc_object.name))
                outf.write('    <pose>{}</pose>'.format(voc_object.pose))
                outf.write('    <truncated>{}</truncated>'.format(voc_object.truncated))
                outf.write('    <difficult>{}</difficult>'.format(voc_object.difficult))
                outf.write('    <bndbox>')
                outf.write('      <xmin>{}</xmin>'.format(voc_object.bndbox.xmin))
                outf.write('      <ymin>{}</ymin>'.format(voc_object.bndbox.ymin))
                outf.write('      <xmax>{}</xmax>'.format(voc_object.bndbox.xmax))
                outf.write('      <ymax>{}</ymax>'.format(voc_object.bndbox.ymax))
                outf.write('    </bndbox>')
                outf.write('  </object>')
        outf.write('</annotation>')

    def load_bbox(self):
        """
        解析测试部分输出的bounding box
        :return: 
        """
        return

    def count_dataset_objects(self, dataset='', silence=True):
        """
        计算数据集中各类别的目标总数。
        读取测试/训练集index文件（.path/to/VOC/ImageSets/test.txt），根据文件解析相应标注文件。
        :param dataset: 选择计算哪部分的目标数量。
                        可选：测试集'test'，训练集'train'，验证集'val'，训练和验证集'trainval'。
                        默认为空，统计全部数据集。
        :param silence: 控制是否隐藏统计过程的控制台输出
        :return: 
        """
        return

    def count_label_objects(self, label, dataset='', silence=True):
        """
        计算数据集中指定label的目标数。
        读取测试/训练集index文件（.path/to/VOC/ImageSets/test.txt），根据文件解析相应标注文件。
        :param label: 需要统计的label名称
        :param dataset: 选择计算哪部分的目标数量。
                        可选：测试集'test'，训练集'train'，验证集'val'，训练和验证集'trainval'。
                        默认为空，统计全部数据集。
        :param silence: 控制是否隐藏统计过程的控制台输出
        :return: label_count: 计算出的目标数量
        """
        import xml.etree.ElementTree as ET
        label_count = 0  # label 计数

        if isinstance(label, int):
            label = str(label)

        # 获得annotation文件名列表
        anno_files = []
        if dataset == '':
            anno_files = os.listdir(self.anno_path)
        elif dataset == 'test':
            if os.path.isfile(self.testset_path):
                test_list = open(self.testset_path)
                for line in test_list:
                    line = line.strip('\n')
                    line = line.strip('\r')
                    anno_files.append(line + '.xml')
                test_list.close()
            else:
                print '\'{}\' test set file not exist.'.format(self.testset_path)
                return -1
        elif dataset == 'val':
            if os.path.isfile(self.valset_path):
                test_list = open(self.valset_path)
                for line in test_list:
                    line = line.strip('\n')
                    line = line.strip('\r')
                    anno_files.append(line + '.xml')
                test_list.close()
            else:
                print '\'{}\' val set file not exist.'.format(self.valset_path)
                return -1
        elif dataset == 'train':
            if os.path.isfile(self.trainset_path):
                test_list = open(self.trainset_path)
                for line in test_list:
                    line = line.strip('\n')
                    line = line.strip('\r')
                    anno_files.append(line + '.xml')
                test_list.close()
            else:
                print '\'{}\' train set file not exist.'.format(self.trainset_path)
                return -1
        elif dataset == 'trainval':
            if os.path.isfile(self.trainvalset_path):
                test_list = open(self.trainvalset_path)
                for line in test_list:
                    line = line.strip('\n')
                    line = line.strip('\r')
                    anno_files.append(line + '.xml')
                test_list.close()
            else:
                print '\'{}\' trainval set file not exist.'.format(self.trainvalset_path)
                return -1

        if not silence:
            print 'start to find label \'{}\':'.format(label)

        for index in anno_files:
            file_path = os.path.join(self.anno_path, index)
            # 解析对应文件
            if os.path.isfile(file_path):
                tree = ET.parse(file_path)
            else:
                print 'Path \'{}\' is invalid.'.format(file_path)
                return
            local_count = 0  # 本文件中对应label的目标计数
            # 从文件中解析所有object标签
            objs = tree.findall('object')
            for ix, obj in enumerate(objs):
                if obj.find('name').text == label:
                    local_count += 1
                    label_count += 1
            if not silence:
                print 'found {} objects in file {}, totally {}.'.format(local_count, index, label_count)
        if not silence:
            print 'found {} \'{}\' objects in {} files.'.format(label_count, label, len(anno_files))
        return label_count

    def missed_detection(self):
        """
        计算漏检和错检数及漏检、错检率。
        :return: 
        """
        return