import React from 'react';
import { Layout, Menu, Button, Badge, Tooltip, Space } from 'antd';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  HomeOutlined,
  AreaChartOutlined,
  SettingOutlined,
  CloudServerOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '@store/useAppStore';
import type { MenuProps } from 'antd';

const { Header, Sider } = Layout;

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { sidebarCollapsed, setSidebarCollapsed, systemStatus } = useAppStore();

  // 菜单项配置
  const menuItems: MenuProps['items'] = [
    {
      key: '/detection',
      icon: <HomeOutlined />,
      label: '变化检测',
    },
    {
      key: '/results',
      icon: <AreaChartOutlined />,
      label: '结果查看',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: '系统设置',
    },
  ];

  const handleMenuClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key);
  };

  // 获取系统状态图标和颜色
  const getStatusIcon = () => {
    if (!systemStatus) {
      return <CloudServerOutlined style={{ color: '#999' }} />;
    }

    switch (systemStatus.status) {
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <CloudServerOutlined style={{ color: '#999' }} />;
    }
  };

  const getStatusText = () => {
    if (!systemStatus) return '连接中...';
    
    switch (systemStatus.status) {
      case 'healthy':
        return '系统正常';
      case 'warning':
        return '系统警告';
      case 'error':
        return '系统异常';
      default:
        return '状态未知';
    }
  };

  return (
    <Layout style={{ height: '100%' }}>
      {/* 侧边栏 */}
      <Sider 
        trigger={null} 
        collapsible 
        collapsed={sidebarCollapsed}
        width={250}
        theme="light"
        style={{
          borderRight: '1px solid #f0f0f0',
          boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
        }}
      >
        {/* Logo区域 */}
        <div 
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: sidebarCollapsed ? 'center' : 'flex-start',
            padding: sidebarCollapsed ? 0 : '0 24px',
            borderBottom: '1px solid #f0f0f0',
            background: '#fff',
          }}
        >
          {sidebarCollapsed ? (
            <div 
              style={{ 
                fontSize: 20, 
                fontWeight: 'bold', 
                color: '#4CAF50' 
              }}
            >
              GIS
            </div>
          ) : (
            <div>
              <div 
                style={{ 
                  fontSize: 16, 
                  fontWeight: 'bold', 
                  color: '#4CAF50',
                  lineHeight: 1.2,
                }}
              >
                地理测绘变化检测
              </div>
              <div 
                style={{ 
                  fontSize: 12, 
                  color: '#999',
                  lineHeight: 1,
                }}
              >
                GIS Change Detection
              </div>
            </div>
          )}
        </div>

        {/* 导航菜单 */}
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ 
            borderRight: 'none',
            background: '#fff',
          }}
        />

        {/* 状态指示器（在底部） */}
        {!sidebarCollapsed && (
          <div 
            style={{
              position: 'absolute',
              bottom: 16,
              left: 16,
              right: 16,
              padding: 12,
              background: '#f8f9fa',
              borderRadius: 6,
              border: '1px solid #e9ecef',
            }}
          >
            <Space size="small">
              {getStatusIcon()}
              <span style={{ fontSize: 12, color: '#666' }}>
                {getStatusText()}
              </span>
            </Space>
            {systemStatus?.services && (
              <div style={{ marginTop: 8, fontSize: 11, color: '#999' }}>
                <div>检测服务: {systemStatus.services.changeDetection}</div>
                <div>文件存储: {systemStatus.services.fileStorage}</div>
              </div>
            )}
          </div>
        )}
      </Sider>

      <Layout>
        {/* 头部 */}
        <Header 
          style={{ 
            padding: '0 16px',
            background: '#fff',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          }}
        >
          {/* 左侧：折叠按钮 */}
          <Button
            type="text"
            icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            style={{
              fontSize: '16px',
              width: 64,
              height: 64,
            }}
          />

          {/* 右侧：状态和操作 */}
          <Space size="large">
            {/* 系统状态指示器 */}
            <Tooltip title={getStatusText()}>
              <Badge 
                status={
                  systemStatus?.status === 'healthy' ? 'success' :
                  systemStatus?.status === 'warning' ? 'warning' :
                  systemStatus?.status === 'error' ? 'error' : 'default'
                }
                text={sidebarCollapsed ? undefined : getStatusText()}
              />
            </Tooltip>

            {/* 统计信息（如果有） */}
            {systemStatus?.statistics && (
              <Space size="middle">
                <Tooltip title="总任务数">
                  <Badge 
                    count={systemStatus.statistics.totalTasks} 
                    showZero 
                    style={{ backgroundColor: '#52c41a' }}
                  >
                    <span style={{ fontSize: 12, color: '#666' }}>任务</span>
                  </Badge>
                </Tooltip>
                
                <Tooltip title={`平均处理时间: ${systemStatus.statistics.averageProcessingTime.toFixed(1)}秒`}>
                  <span style={{ fontSize: 12, color: '#666' }}>
                    {systemStatus.statistics.averageProcessingTime.toFixed(1)}s
                  </span>
                </Tooltip>
              </Space>
            )}
          </Space>
        </Header>

        {/* 主要内容区域 */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {children}
        </div>
      </Layout>
    </Layout>
  );
};

export default MainLayout;