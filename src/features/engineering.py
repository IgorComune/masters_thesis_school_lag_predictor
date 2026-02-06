import pandas as pd


class LineAverager:
    """Compute the row-wise average of a DataFrame.

    This class adds a new column to a pandas DataFrame containing
    the mean of each row, regardless of the number of columns.
    """

    def __init__(self, target_col: str = "media") -> None:
        """Initialize the LineAverager.

        Args:
            target_col (str): Name of the column that will store
                the row-wise average.
        """
        self.target_col = target_col

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a column with the row-wise average to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The same DataFrame with an additional
            column containing the row-wise average.
        """
        df[self.target_col] = df.mean(axis=1)
        return df
