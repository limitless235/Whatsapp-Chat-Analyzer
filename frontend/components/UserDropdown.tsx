import React from "react";

export type UserDropdownProps = {
  users: string[];
  selectedUser: string | null;
  onChange: (value: string) => void;
};

const UserDropdown: React.FC<UserDropdownProps> = ({ users, selectedUser, onChange }) => {
  return (
    <div className="my-4">
      <label htmlFor="user-dropdown" className="block font-semibold mb-2">
        Select User
      </label>
      <select
        id="user-dropdown"
        className="p-2 border border-gray-300 rounded"
        value={selectedUser || ""}
        onChange={(e) => onChange(e.target.value)}
      >
        {users.map((user) => (
          <option key={user} value={user}>
            {user}
          </option>
        ))}
      </select>
    </div>
  );
};

export default UserDropdown;
