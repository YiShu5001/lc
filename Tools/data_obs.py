class DataTracker:
    def __init__(self):
        self.data = []  # 存储所有不重复的数据
        self.categories = {}  # 存储数据与类别的对应关系

    def add_data(self, item):
        """添加数据，如果未存储过则保留"""
        if item not in self.data:
            self.data.append(item)
            return True  # 表示成功添加新数据
        return False  # 表示数据已存在

    def show_all_data(self):
        """显示所有存储的数据"""
        print("\n所有已存储的数据情况：")
        if not self.data:
            print("暂无数据")
            return
        for i, item in enumerate(self.data, 1):
            print(f"{i}. {item}")

    def set_category(self, data_item, category):
        """为数据项设置类别"""
        if data_item in self.data:
            self.categories[data_item] = category
            return True
        return False

    def show_categories(self):
        """显示数据与类别的对应关系"""
        print("\n数据与类别的对应关系：")
        if not self.categories:
            print("暂无类别设置")
            return
        for data_item, category in self.categories.items():
            print(f"{data_item} -> {category}")

    def get_category_mapping(self):
        """获取类别到数据的映射"""
        category_map = {}
        for data_item, category in self.categories.items():
            if category not in category_map:
                category_map[category] = []
            category_map[category].append(data_item)
        return category_map

    def show_category_groups(self):
        """按类别分组显示数据"""
        print("\n按类别分组的数据：")
        category_map = self.get_category_mapping()
        if not category_map:
            print("暂无类别设置")
            return
        for category, items in category_map.items():
            print(f"{category}: {', '.join(items)}")


def main():
    tracker = DataTracker()

    while True:
        print("\n===== 数据情况追踪工具 =====")
        print("1. 输入新数据")
        print("2. 查看所有数据情况")
        print("3. 为数据设置类别")
        print("4. 查看数据-类别对应关系")
        print("5. 查看按类别分组的数据")
        print("6. 退出")

        choice = input("请选择操作 (1-6): ")

        if choice == '1':
            item = input("请输入数据: ")
            if tracker.add_data(item):
                print(f"已添加新数据: {item}")
            else:
                print(f"数据 '{item}' 已存在，未重复添加")

        elif choice == '2':
            tracker.show_all_data()

        elif choice == '3':
            if not tracker.data:
                print("请先添加数据")
                continue
            item = input("请输入要设置类别的数据: ")
            category = input("请输入类别: ")
            if tracker.set_category(item, category):
                print(f"已将 '{item}' 归类为 '{category}'")
            else:
                print(f"数据 '{item}' 不存在")

        elif choice == '4':
            tracker.show_categories()

        elif choice == '5':
            tracker.show_category_groups()

        elif choice == '6':
            print("感谢使用，再见！")
            break

        else:
            print("无效的选择，请重试")


if __name__ == "__main__":
    main()
