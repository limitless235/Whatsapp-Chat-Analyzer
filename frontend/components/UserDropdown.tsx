"use client";

import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

type UserDropdownProps = {
  options: string[];
  selected: string;
  onChange: (value: string) => void;
  placeholder?: string;
};

const UserDropdown: React.FC<UserDropdownProps> = ({
  options,
  selected,
  onChange,
  placeholder = "Select a user",
}) => {
  return (
    <Select value={selected} onValueChange={onChange}>
      <SelectTrigger className="w-[250px]">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option} value={option}>
            {option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default UserDropdown;
