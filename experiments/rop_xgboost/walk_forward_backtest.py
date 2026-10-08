"""
Walk-Forward Inventory Replenishment Backtest Engine
===================================================
Simulates operational inventory replenishment over a rolling historical horizon.
Strictly leakage-free: at each decision date t, only historical observations known at t
are visible to the policy.

Evaluates operational KPIs:
- Stockout rate (%)
- Unit fill rate / Cycle service level (%)
- Average inventory level & Days of Cover
- Reorder frequency (number of purchase orders triggered)
- Total units ordered
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from experiments.rop_xgboost.config import (
    DEFAULT_LEAD_TIME_DAYS, MAX_COVER_DAYS_TARGET
)
from experiments.rop_xgboost.rop_target import (
    compute_reorder_trigger, compute_reorder_quantity
)


class ReplenishmentSimulator:
    """Simulates daily inventory transitions and order placement over a backtest horizon."""

    def __init__(
        self,
        lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
        target_cover_days: int = MAX_COVER_DAYS_TARGET,
        initial_stock_cover_days: int = 14
    ):
        self.lead_time_days = lead_time_days
        self.target_cover_days = target_cover_days
        self.initial_stock_cover_days = initial_stock_cover_days

    def run_simulation(
        self,
        df_backtest: pd.DataFrame,
        model,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Runs a day-by-day discrete-event inventory simulation across all series.
        """
        print(f"\n[BACKTEST SIMULATION] Running walk-forward backtest for: {model_name}...")
        print(f"  Lead Time: {self.lead_time_days} days | Target Cover: {self.target_cover_days} days")

        # Generate ROP predictions for the entire backtest dataset
        df_sim = df_backtest.copy().sort_values(['canonical_sku', 'platform_group', 'date'])
        df_sim['predicted_rop'] = model.predict_rop(df_sim)

        unique_series = df_sim[['canonical_sku', 'platform_group']].drop_duplicates().values
        dates = sorted(df_sim['date'].unique())

        total_demand_units = 0.0
        total_fulfilled_units = 0.0
        total_stockout_events = 0
        total_active_days = 0
        total_pos_triggered = 0
        total_units_ordered = 0.0
        daily_inventory_samples = []
        daily_doc_samples = []

        # Iterate per series to maintain clean inventory state
        for sku, platform in unique_series:
            s_data = df_sim[(df_sim['canonical_sku'] == sku) & (df_sim['platform_group'] == platform)].set_index('date')
            
            # Initial stock: use actual starting stock or baseline run rate cover
            first_row = s_data.iloc[0]
            run_rate = max(float(first_row.get('v14', 0.1)), 0.05)
            curr_stock = float(first_row.get('current_stock', run_rate * self.initial_stock_cover_days))
            if curr_stock <= 0:
                curr_stock = run_rate * self.initial_stock_cover_days

            in_transit_orders = {}  # arrival_date -> quantity

            for current_date in dates:
                if current_date not in s_data.index:
                    continue

                row = s_data.loc[current_date]
                actual_demand = float(row.get('observed_units_sold', 0.0))
                rop = float(row.get('predicted_rop', 1.0))
                daily_v = max(float(row.get('v14', run_rate)), 0.05)

                # 1. Inbound PO deliveries arrive at start of day
                if current_date in in_transit_orders:
                    curr_stock += in_transit_orders.pop(current_date)

                # 2. Demand fulfillment
                total_demand_units += actual_demand
                if actual_demand > 0:
                    total_active_days += 1
                    if curr_stock >= actual_demand:
                        total_fulfilled_units += actual_demand
                        curr_stock -= actual_demand
                    else:
                        # Stockout occurs
                        total_fulfilled_units += curr_stock
                        total_stockout_events += 1
                        curr_stock = 0.0

                # Track inventory and cover
                daily_inventory_samples.append(curr_stock)
                doc = curr_stock / daily_v if daily_v > 0 else 99.0
                daily_doc_samples.append(min(doc, 90.0))

                # 3. Replenishment decision at end of day
                # Inventory Position = On-Hand + Total In-Transit
                pipeline_qty = sum(in_transit_orders.values())
                inventory_position = curr_stock + pipeline_qty

                if compute_reorder_trigger(inventory_position, rop):
                    roq = compute_reorder_quantity(
                        inventory_position=inventory_position,
                        rop=rop,
                        daily_run_rate=daily_v,
                        target_cover_days=self.target_cover_days
                    )
                    if roq > 0:
                        # Schedule delivery L days later
                        arr_idx = dates.index(current_date) + self.lead_time_days
                        if arr_idx < len(dates):
                            arr_date = dates[arr_idx]
                            in_transit_orders[arr_date] = in_transit_orders.get(arr_date, 0.0) + roq
                        total_pos_triggered += 1
                        total_units_ordered += roq

        # Aggregate backtest metrics
        service_level_pct = (total_fulfilled_units / total_demand_units * 100.0) if total_demand_units > 0 else 100.0
        stockout_rate_pct = (total_stockout_events / total_active_days * 100.0) if total_active_days > 0 else 0.0
        mean_inventory = float(np.mean(daily_inventory_samples)) if daily_inventory_samples else 0.0
        mean_doc = float(np.mean(daily_doc_samples)) if daily_doc_samples else 0.0

        metrics = {
            'model_name': model_name,
            'lead_time_days': self.lead_time_days,
            'total_demand_units': round(total_demand_units, 1),
            'total_fulfilled_units': round(total_fulfilled_units, 1),
            'unit_service_level_pct': round(service_level_pct, 2),
            'stockout_rate_pct': round(stockout_rate_pct, 2),
            'total_stockout_events': total_stockout_events,
            'total_pos_triggered': total_pos_triggered,
            'total_units_ordered': round(total_units_ordered, 1),
            'average_inventory_units': round(mean_inventory, 1),
            'average_days_of_cover': round(mean_doc, 1)
        }

        print(f"  --> Service Level: {service_level_pct:.2f}% | Stockout Rate: {stockout_rate_pct:.2f}%")
        print(f"  --> Avg Inventory: {mean_inventory:.1f} units | POs Triggered: {total_pos_triggered}")
        return metrics
