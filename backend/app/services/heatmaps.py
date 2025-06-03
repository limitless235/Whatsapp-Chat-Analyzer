from typing import List, Dict, Any
import pandas as pd

class HeatmapService:
    def __init__(self):
        pass

    def _parse_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
    
        df = pd.DataFrame(messages)

        if "date" not in df.columns:
            raise ValueError("Missing 'timestamp' field in message data.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        if df.empty:
            raise ValueError("No valid timestamps found after parsing.")

        df["day"] = df["date"].dt.day_name()
        df["hour"] = df["date"].dt.hour
        df["date"] = df["date"].dt.date
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
        pivot = df.pivot_table(index="sender_name", columns="hour", values="text", aggfunc="count", fill_value=0)

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
        daily_counts = df.groupby(["sender_name", "date"]).size().unstack(fill_value=0)

        return {
            "x": daily_counts.columns.astype(str).tolist(),  # dates as strings
            "y": daily_counts.index.tolist(),                # users
            "z": daily_counts.values.tolist()                # msg counts per user per date
        }
def generate_heatmaps(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Top-level wrapper that computes all heatmap-related data.
    Expects a cleaned DataFrame and returns a dictionary of heatmap results.
    """
    messages = df.to_dict(orient="records")
    service = HeatmapService()

    return {
        "chat_activity": service.chat_activity_heatmap(messages),
        "user_activity": service.user_activity_heatmap(messages),
        "user_daily_counts": service.user_daily_message_counts(messages),
    }
