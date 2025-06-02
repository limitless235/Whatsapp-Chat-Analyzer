"use client";

import React from "react";

type PlotWrapperProps = {
  title: string;
  children: React.ReactNode;
  className?: string;
};

const PlotWrapper: React.FC<PlotWrapperProps> = ({ title, children, className = "" }) => {
  return (
    <div className={`bg-white dark:bg-gray-900 rounded-2xl shadow-md p-4 mb-6 ${className}`}>
      <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">{title}</h2>
      <div className="overflow-x-auto">{children}</div>
    </div>
  );
};

export default PlotWrapper;
