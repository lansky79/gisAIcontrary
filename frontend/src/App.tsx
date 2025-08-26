import React, { useEffect } from 'react';
import { Layout, message } from 'antd';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from '@components/Layout/MainLayout';
import DetectionPage from '@components/Pages/DetectionPage';
import ResultsPage from '@components/Pages/ResultsPage';
import SettingsPage from '@components/Pages/SettingsPage';
import { useAppStore } from '@store/useAppStore';
import { healthCheckApi } from '@services/api';

const { Content } = Layout;

const App: React.FC = () => {
  const { setSystemStatus } = useAppStore();

  // 系统健康检查
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await healthCheckApi();
        if (response.success && response.data) {
          setSystemStatus(response.data);
        }
      } catch (error) {
        console.error('系统健康检查失败:', error);
        message.error('系统服务连接失败，请检查后端服务是否正常运行');
      }
    };

    // 立即检查一次
    checkSystemHealth();

    // 每30秒检查一次
    const interval = setInterval(checkSystemHealth, 30000);

    return () => clearInterval(interval);
  }, [setSystemStatus]);

  return (
    <Router>
      <Layout style={{ height: '100vh' }}>
        <MainLayout>
          <Content style={{ height: '100%', overflow: 'hidden' }}>
            <Routes>
              <Route path="/" element={<DetectionPage />} />
              <Route path="/detection" element={<DetectionPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </Content>
        </MainLayout>
      </Layout>
    </Router>
  );
};

export default App;