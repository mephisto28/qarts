

if __name__ == "__main__":
    from qarts.core import PanelBlockIndexed, PanelBlockDense
    from qarts.loader import ParquetPanelLoader

    loader = ParquetPanelLoader()
    block = loader.load_intraday_quotation(date='20230104')
    block_dense = PanelBlockDense.from_indexed_block(block, required_columns=['1min_v4_barra4_total'], fill_methods=[1])
    print(block_dense.data.shape)
    breakpoint()