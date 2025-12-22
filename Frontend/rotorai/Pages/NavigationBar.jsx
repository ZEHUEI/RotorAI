import React from 'react';

const NavigationBar = () => {
  return (
    <>
      <div className="fixed top-0 left-0 right-0 z-50 w-full backdrop-blur-lg overflow-hidden text-2xl">
        <div className="flex flex-row gap-14 px-1.5 md:px-20 lg:px-40 xl:px-40 2xl:px-40 py-4 font-[AT]">
          <div>Logo</div>
          <div>About</div>
          <div>Blog</div>
        </div>
      </div>
    </>
  );
};

export default NavigationBar;
