from typing import List, Dict, Any
import pandas as pd

class HeatmapService:
    def __init__(self):
        pass

    def _parse_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert raw message dicts to a DataFrame with datetime.
        """
        df = pd.DataFrame(messages)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)

        # Extract time components
        df["day"] = df["timestamp"].dt.day_name()
        df["hour"] = df["timestamp"].dt.hour
        df["date"] = df["timestamp"].dt.date
        return df

    def chat_activity_heatmap(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heatmap data of overall activity by day and hour.
        """
        df = self._parse_dataframe(messages)
        pivot = df.pivot_table(index="day", columns="hour", values="text", aggfunc="count", fill_value=0)

        # Ensure order of days
        ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(ordered_days)

        return {
            "x": pivot.columns.tolist(),   # hours
            "y": pivot.index.tolist(),     # days
            "z": pivot.values.tolist()     # counts
        }

    def user_activity_heatmap(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heatmap of activity: rows = users, columns = hours, values = message count
        """
        df = self._parse_dataframe(messages)
        pivot = df.pivot_table(index="sender", columns="hour", values="text", aggfunc="count", fill_value=0)

        return {
            "x": pivot.columns.tolist(),   # hours
            "y": pivot.index.tolist(),     # senders
            "z": pivot.values.tolist()     # counts
        }

    def user_daily_message_counts(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Time-series message count per user by date (for line plots or time heatmaps).
        """
        df = self._parse_dataframe(messages)
        daily_counts = df.groupby(["sender", "date"]).size().unstack(fill_value=0)

        return {
            "x": daily_counts.columns.astype(str).tolist(),  # dates as strings
            "y": daily_counts.index.tolist(),                # users
            "z": daily_counts.values.tolist()                # msg counts per user per date
        }
